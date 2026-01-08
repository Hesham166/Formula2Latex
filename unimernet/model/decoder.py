import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from .components import SqueezeAttention


class UniMERNetDecoderLayer(nn.Module):
    """
    Transformer decoder layer with squeeze attention.
    
    Architecture:
        Self-Attention -> Cross-Attention -> Feed-Forward

    Args:
        embed_dim: Input/Output dimension.
        num_heads: Number of attention heads.
        ffn_dim: Hidden dimension for the feed-forward network.
        dropout: Dropout probability.
        qk_squeeze: Squeeze factor for query and key.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        qk_squeeze: int = 2
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.self_attn = SqueezeAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qk_squeeze=1,               # No squeezing for self-attention
            dropout=dropout,
            is_decoder=True
        )

        self.cross_attn = SqueezeAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qk_squeeze=qk_squeeze,
            dropout=dropout,
            is_decoder=True
        )

        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.act = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states: Decoder input (B, L, D)
            encoder_hidden_states: Encoder output for cross-attention (B, N, D)
            encoder_attention_mask: Mask for encoder outputs
            past_key_value: Cached (self_kv, cross_kv) from previous steps
            attention_mask: Causal mask for self-attention
            
        Returns:
            output: Layer output (B, L, D)
            present_kv: Updated cache ((self_k, self_v), (cross_k, cross_v))
        """
        self_attn_past = past_key_value[0] if past_key_value is not None else None
        cross_attn_past = past_key_value[1] if past_key_value is not None else None

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        hidden_states, present_self = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past,
            attention_mask=attention_mask
        )
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        present_cross = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.norm2(hidden_states)
            
            hidden_states, present_cross = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                past_key_value=cross_attn_past,
                attention_mask=encoder_attention_mask
            )
            
            hidden_states = self.dropout(hidden_states)
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = residual + hidden_states

        present_key_value = (present_self, present_cross)

        return hidden_states, present_key_value


class UniMERNetDecoder(nn.Module):
    """
    Autoregressive decoder for LaTeX sequence generation.
    
    Args:
        vocab_size: Size of token vocabulary
        max_position_embeddings: Maximum sequence length
        embed_dim: Embedding dimension
        num_layers: Number of decoder layers
        num_heads: Number of attention heads
        ffn_dim: Feed-forward dimension
        dropout: Dropout rate
        qk_squeeze: Squeeze factor for cross-attention
        pad_token_id: Padding token ID
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        qk_squeeze: int = 2,
        pad_token_id: int = 1,
    ):
        super().__init__()

        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.embed_positions = nn.Embedding(max_position_embeddings, embed_dim)

        self.layers = nn.ModuleList([
            UniMERNetDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                qk_squeeze=qk_squeeze
            )
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.embed_scale = embed_dim ** 0.5

    def get_input_embeddings(self) -> nn.Embedding:
        """Get token embedding layer."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        """Set token embedding layer."""
        self.embed_tokens = value

    def _prepare_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Create causal mask for autoregressive generation.
        
        Returns lower-triangular mask (allows attending to past, not future).
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        """
        Args:
            input_ids: Input token IDs (B, L)
            encoder_hidden_states: Encoded visual features (B, N, D)
            encoder_attention_mask: Mask for encoder outputs
            past_key_values: Cached key-values from previous steps
            use_cache: Whether to return updated cache
            
        Returns:
            If use_cache=False: logits (B, L, vocab_size)
            If use_cache=True: (logits, updated_cache)
        """
        B, L = input_ids.shape
        device = input_ids.device

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            start_pos = past_key_values[0][0][0].shape[2]
        else:
            start_pos = 0

        current_seq_len = input_ids.shape[1]
        positions = torch.arange(start_pos, start_pos + current_seq_len, device=device).unsqueeze(0)
        
        x = self.embed_tokens(input_ids) * self.embed_scale
        x = x + self.embed_positions(positions)
        x = self.dropout(x)

        if past_key_values is None:
            causal_mask = self._prepare_causal_mask(L, device, x.dtype)
        else:
            causal_mask = None
        
        next_cache = () if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            x, layer_cache = layer(
                hidden_states=x,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=layer_past,
                attention_mask=causal_mask
            )
            
            if use_cache:
                next_cache += (layer_cache,)

        x = self.layer_norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, next_cache
        return logits