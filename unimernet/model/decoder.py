import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from .components import SqueezeAttention


class UniMERNetDecoderLayer(nn.Module):
    """Transformer decoder layer with squeeze attention."""
    
    def __init__(
        self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, qk_squeeze: int = 2
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = SqueezeAttention(embed_dim, num_heads, qk_squeeze=1, dropout=dropout, is_decoder=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = SqueezeAttention(embed_dim, num_heads, qk_squeeze=qk_squeeze, dropout=dropout, is_decoder=True)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor]]]:
        
        self_attn_past = past_key_value[0] if past_key_value is not None else None
        cross_attn_past = past_key_value[1] if past_key_value is not None else None

        residual = hidden_states

        hidden_states, present_self = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past,
            attention_mask=attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.norm1(hidden_states)

        present_cross = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            
            hidden_states, present_cross = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                past_key_value=cross_attn_past,
                attention_mask=encoder_attention_mask
            )
            hidden_states = self.dropout(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = self.norm2(hidden_states)

        residual = hidden_states
        hidden_states = self.act(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.norm3(hidden_states)

        present_key_value = (present_self, present_cross)

        return hidden_states, present_key_value


class UniMERNetDecoder(nn.Module):
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
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.embed_scale = embed_dim ** 0.5

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def _prepare_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True
    ) -> Union[torch.Tensor, Tuple]:
        
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
            past = past_key_values[i] if past_key_values is not None else None
            
            x, layer_cache = layer(
                hidden_states=x,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past,
                attention_mask=causal_mask
            )
            
            if use_cache:
                next_cache += (layer_cache,)

        x = self.layer_norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, next_cache
        return logits
