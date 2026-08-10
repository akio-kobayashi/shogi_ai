import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def prepare_sdpa_mask(attention_mask: Optional[torch.Tensor]):
    """2D padding maskを全層で共有するcausal SDPA maskへ一度だけ変換する。"""
    if attention_mask is None or attention_mask.ndim == 4:
        return attention_mask
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, seq_len]")
    seq_len = attention_mask.shape[1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device).tril()
    return causal[None, None] & attention_mask.to(torch.bool)[:, None, None, :]


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

    def _attention(self, q, k, v, attention_mask=None):
        batch, _, seq_len, _ = q.shape
        if attention_mask is None:
            mask = None
            is_causal = True
        else:
            mask = prepare_sdpa_mask(attention_mask)
            is_causal = False
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

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

        output = self._attention(q, k, v, attention_mask)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.resid_dropout(self.out_proj(output))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """full causal forwardと同時にprefix用K/Vを返す。

        trace生成のVanilla prefill専用。通常のforward_stepをtokenごとに
        呼ぶより速く、K/V projectionを後から再計算する必要もない。
        """
        batch, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        output = self._attention(q, k, v, attention_mask)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.resid_dropout(self.out_proj(output))
        return output, (k, v)

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

    def forward_with_cache(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Vanilla prefill用。block出力とattention K/Vを同時に返す。"""
        attention, key_value = self.attn.forward_with_cache(
            self.attn_norm(x), attention_mask
        )
        x = x + attention
        return x + self.ffn(self.ffn_norm(x)), key_value
