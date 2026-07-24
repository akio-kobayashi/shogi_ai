import math
from typing import Optional, Tuple

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(dtype=x.dtype)) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=x.device
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal.view(1, 1, seq_len, seq_len), float("-inf"))
        if attention_mask is not None:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError("attention_mask must have shape [batch, seq_len]")
            invalid_keys = ~attention_mask.to(dtype=torch.bool)
            scores = scores.masked_fill(
                invalid_keys[:, None, None, :], float("-inf")
            )

        weights = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        weights = self.attn_dropout(weights)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.resid_dropout(self.out_proj(output))

    def forward_step(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """paddingのない逐次評価用KV-cache forward。"""
        if x.shape[1] != 1:
            raise ValueError("forward_step expects exactly one token")
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().view(x.shape[0], 1, self.d_model)
        return self.resid_dropout(self.out_proj(output)), (k, v)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, eps: float):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model, eps)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask)
        return x + self.ffn(self.ffn_norm(x))

    def forward_step(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attention, key_value = self.attn.forward_step(
            self.attn_norm(x), past_key_value
        )
        x = x + attention
        return x + self.ffn(self.ffn_norm(x)), key_value
