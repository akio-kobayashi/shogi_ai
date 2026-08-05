"""小規模将棋実験用のLLaMA型 causal decoder。

既学習LLMを流用するものではない。RoPE，Pre-RMSNorm，SwiGLUを備えた
decoder-only Transformerを，プロジェクト固有語彙からランダム初期化して学習する。
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from .config import ModelConfig
from .layers import RMSNorm
from .outputs import DecoderOutput


KeyValue = Tuple[torch.Tensor, torch.Tensor]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryCausalSelfAttention(nn.Module):
    """RoPEをQ/Kへ適用するmulti-head causal attention。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float, rope_theta: float = 10000.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.rope_theta = float(rope_theta)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, position_offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        length = q.shape[-2]
        positions = torch.arange(position_offset, position_offset + length, device=q.device, dtype=torch.float32)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=q.device, dtype=torch.float32) / self.head_dim))
        angles = torch.outer(positions, inv_freq)
        angles = torch.cat((angles, angles), dim=-1)
        cos = angles.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=q.dtype)[None, None, :, :]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

    def _project(self, x: torch.Tensor, position_offset: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        q, k = self._apply_rope(q, k, position_offset)
        return q, k, v

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_offset: int = 0) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, k, v = self._project(x, position_offset)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(diagonal=1)
        scores = scores.masked_fill(causal[None, None], float("-inf"))
        if attention_mask is not None:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError("attention_mask must have shape [batch, seq_len]")
            scores = scores.masked_fill(~attention_mask.to(dtype=torch.bool)[:, None, None, :], float("-inf"))
        weights = self.attn_dropout(torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype))
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.resid_dropout(self.out_proj(output))

    def forward_step(self, x: torch.Tensor, past_key_value: Optional[KeyValue], position: int) -> Tuple[torch.Tensor, KeyValue]:
        if x.shape[1] != 1:
            raise ValueError("forward_step expects exactly one token")
        q, k, v = self._project(x, position)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k, v = torch.cat((past_k, k), dim=2), torch.cat((past_v, v), dim=2)
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        output = torch.matmul(weights.to(dtype=q.dtype), v).transpose(1, 2).contiguous().view(x.shape[0], 1, self.d_model)
        return self.resid_dropout(self.out_proj(output)), (k, v)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)))


class LlamaDecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = RotaryCausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask)
        return x + self.ffn(self.ffn_norm(x))

    def forward_step(self, x: torch.Tensor, past_key_value: Optional[KeyValue], position: int) -> Tuple[torch.Tensor, KeyValue]:
        attention, key_value = self.attn.forward_step(self.attn_norm(x), past_key_value, position)
        x = x + attention
        return x + self.ffn(self.ffn_norm(x)), key_value


class LlamaTransformer(nn.Module):
    """RoPE + Pre-RMSNorm + SwiGLUのLLaMA型decoder。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(LlamaDecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, input_ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if position_offset + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        return self.embedding_dropout(self.token_embedding(input_ids))

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, recurrent_mask: Optional[torch.Tensor] = None, exact_recurrence: bool = False) -> DecoderOutput:
        del recurrent_mask
        if exact_recurrence:
            return self.forward_exact(input_ids)
        x = self._embed(input_ids)
        hidden_states = [x]
        for layer in self.layers:
            x = layer(x, attention_mask)
            hidden_states.append(x)
        return DecoderOutput(logits=self.lm_head(self.final_norm(x)), hidden_states=tuple(hidden_states))

    def step(self, input_ids: torch.Tensor, position: int, past_key_values: Optional[Sequence[Optional[KeyValue]]] = None, recurrent_state=None, recurrent_active=None, return_logits: bool = True):
        del recurrent_state, recurrent_active
        if input_ids.shape[1] != 1:
            raise ValueError("step expects input_ids with shape [batch, 1]")
        if past_key_values is None:
            past_key_values = [None] * self.config.n_layers
        x = self._embed(input_ids, position)
        layer_states = [x]
        next_key_values: List[KeyValue] = []
        for layer, past in zip(self.layers, past_key_values):
            x, key_value = layer.forward_step(x, past, position)
            next_key_values.append(key_value)
            layer_states.append(x)
        logits = self.lm_head(self.final_norm(x)) if return_logits else None
        return logits, tuple(next_key_values), None, tuple(layer_states), None

    def forward_exact(self, input_ids: torch.Tensor, recurrent_mask: Optional[torch.Tensor] = None) -> DecoderOutput:
        del recurrent_mask
        logits_by_step: List[torch.Tensor] = []
        states_by_layer: List[List[torch.Tensor]] = [[] for _ in range(self.config.n_layers + 1)]
        key_values = None
        for position in range(input_ids.shape[1]):
            logits, key_values, _, states, _ = self.step(input_ids[:, position : position + 1], position, key_values)
            logits_by_step.append(logits)
            for layer_index, state in enumerate(states):
                states_by_layer[layer_index].append(state)
        return DecoderOutput(logits=torch.cat(logits_by_step, dim=1), hidden_states=tuple(torch.cat(states, dim=1) for states in states_by_layer))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
