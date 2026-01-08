import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
import torch.nn.functional as F

from .encoder import UniMERNetEncoder
from .decoder import UniMERNetDecoder


class UniMERNet(nn.Module):
    """
    UniMERNet: Universal Mathematical Expression Recognition Network.
    
    Args:
        encoder_depths: Number of blocks per encoder stage
        encoder_embed_dim: Base embedding dimension for encoder
        encoder_num_heads: Number of attention heads per stage (int or list)
        encoder_window_size: Window size for window attention
        
        decoder_num_layers: Number of decoder layers
        decoder_num_heads: Number of decoder attention heads
        decoder_embed_dim: Decoder embedding dimension
        decoder_ffn_dim: Decoder feed-forward dimension
        decoder_qk_squeeze: Squeeze factor for decoder cross-attention
        
        vocab_size: Size of LaTeX token vocabulary
        max_seq_len: Maximum output sequence length
        bos_token_id: Begin-of-sequence token ID
        pad_token_id: Padding token ID
        eos_token_id: End-of-sequence token ID
        
        drop_path_rate: Maximum stochastic depth rate
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        # Encoder configuration
        encoder_depths: List[int] = [6, 6, 6, 6],
        encoder_embed_dim: int = 128,
        encoder_num_heads: Union[int, List[int]] = [4, 8, 16, 32],
        encoder_window_size: int = 7,

        # Decoder configuration
        decoder_num_layers: int = 8,
        decoder_num_heads: int = 8,
        decoder_embed_dim: int = 512,
        decoder_ffn_dim: int = 2048,
        decoder_qk_squeeze: int = 2,

        # Vocabulary and sequence
        vocab_size: int = 50000,
        max_seq_len: int = 1536,
        bos_token_id: int = 0,
        pad_token_id: int = 1,
        eos_token_id: int = 2,
        
        # Regularization
        drop_path_rate: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        self.encoder = UniMERNetEncoder(
            depths=encoder_depths,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            window_size=encoder_window_size,
            drop_path_rate=drop_path_rate,
            dropout=dropout
        )

        self.encoder_output_dim = int(
            encoder_embed_dim * 2 ** (len(encoder_depths) - 1)
        )

        if self.encoder_output_dim != decoder_embed_dim:
            self.enc_to_dec_proj = nn.Linear(
                self.encoder_output_dim, 
                decoder_embed_dim
            )
        else:
            self.enc_to_dec_proj = nn.Identity()

        self.decoder = UniMERNetDecoder(
            vocab_size=vocab_size,
            max_position_embeddings=max_seq_len,
            embed_dim=decoder_embed_dim,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            ffn_dim=decoder_ffn_dim,
            dropout=dropout,
            qk_squeeze=decoder_qk_squeeze,
            pad_token_id=pad_token_id
        )
        
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to visual feature representations.
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            features: Encoded features (B, N_patches, decoder_dim)
        """
        visual_features = self.encoder(images)
        return self.enc_to_dec_proj(visual_features)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        """
        Forward pass for training or cached inference.
        
        Args:
            images: Input images (B, 3, H, W)
            input_ids: Target tokens (B, L) - for training, these are teacher-forced
            encoder_attention_mask: Optional mask for encoder outputs
            past_key_values: Cached key-values for fast inference
            use_cache: Whether to return key-value cache
            
        Returns:
            If use_cache=False: logits (B, L, vocab_size)
            If use_cache=True: (logits, cache) tuple
            
        Note:
            During training, use_cache should be False and input_ids should
            contain the full target sequence shifted right by one position
            (with BOS prepended).
        """
        encoder_hidden_states = self.encode(images)

        result = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        if use_cache:
            logits, cache = result
            return logits, cache
        else:
            return result

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: int = 1536,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Greedy or sampling-based autoregressive generation.
        
        Args:
            images: Input images (B, 3, H, W)
            max_len: Maximum sequence length to generate
            temperature: Sampling temperature (1.0 = no change, <1 = sharper)
            top_k: If set, only sample from top-k tokens
            top_p: If set, nucleus sampling threshold
            
        Returns:
            generated_ids: Generated token sequences (B, L) including BOS, up to EOS or max_len
        """
        B = images.shape[0]
        device = images.device
        
        encoder_hidden_states = self.encode(images)

        input_ids = torch.full(
            (B, 1), 
            self.bos_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        past_key_values = None
        
        for step in range(max_len):
            logits, past_key_values = self.decoder(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, top_k
                )[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            if temperature == 1.0 and top_k is None and top_p is None:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            is_eos = (next_token.squeeze(-1) == self.eos_token_id)
            finished = finished | is_eos
            
            if finished.all():
                break
                
        return input_ids

    def get_encoder_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        return self.encoder_output_dim
    
    def get_num_encoder_params(self) -> int:
        """Count parameters in encoder."""
        return sum(p.numel() for p in self.encoder.parameters())
    
    def get_num_decoder_params(self) -> int:
        """Count parameters in decoder."""
        return sum(p.numel() for p in self.decoder.parameters())
    
    def get_num_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())



def unimernet_tiny(**kwargs) -> UniMERNet:
    """UniMERNet-Tiny: ~100M parameters."""
    defaults = {
        'encoder_depths': [6, 6, 6, 6],
        'encoder_embed_dim': 64,
        'encoder_num_heads': [2, 4, 8, 16],
        'decoder_num_layers': 8,
        'decoder_embed_dim': 512,
        'decoder_ffn_dim': 2048,
    }
    defaults.update(kwargs)
    return UniMERNet(**defaults)


def unimernet_small(**kwargs) -> UniMERNet:
    """UniMERNet-Small: ~200M parameters."""
    defaults = {
        'encoder_depths': [6, 6, 6, 6],
        'encoder_embed_dim': 96,
        'encoder_num_heads': [3, 6, 12, 24],
        'decoder_num_layers': 8,
        'decoder_embed_dim': 768,
        'decoder_ffn_dim': 3072,
    }
    defaults.update(kwargs)
    return UniMERNet(**defaults)


def unimernet_base(**kwargs) -> UniMERNet:
    """UniMERNet-Base: ~325M parameters."""
    defaults = {
        'encoder_depths': [6, 6, 6, 6],
        'encoder_embed_dim': 128,
        'encoder_num_heads': [4, 8, 16, 32],
        'decoder_num_layers': 8,
        'decoder_embed_dim': 1024,
        'decoder_ffn_dim': 4096,
    }
    defaults.update(kwargs)
    return UniMERNet(**defaults)