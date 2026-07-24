from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from .config import ModelConfig
from .layers import DecoderBlock, RMSNorm
from .outputs import DecoderOutput


KeyValue = Tuple[torch.Tensor, torch.Tensor]


class VanillaTransformer(nn.Module):
    """本実験のparameter/data matched基準となるcausal decoder。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            DecoderBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
                config.norm_eps,
            )
            for _ in range(config.n_layers)
        )
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
        seq_len = input_ids.shape[1]
        if position_offset + seq_len > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        positions = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=input_ids.device,
        )
        return self.embedding_dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        recurrent_mask: Optional[torch.Tensor] = None,
        exact_recurrence: bool = False,
    ) -> DecoderOutput:
        del recurrent_mask
        if exact_recurrence:
            return self.forward_exact(input_ids)
        x = self._embed(input_ids)
        hidden_states = [x]
        for layer in self.layers:
            x = layer(x, attention_mask)
            hidden_states.append(x)
        logits = self.lm_head(self.final_norm(x))
        return DecoderOutput(logits=logits, hidden_states=tuple(hidden_states))

    def step(
        self,
        input_ids: torch.Tensor,
        position: int,
        past_key_values: Optional[Sequence[Optional[KeyValue]]] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        recurrent_active: Optional[torch.Tensor] = None,
    ):
        del recurrent_state, recurrent_active
        if input_ids.shape[1] != 1:
            raise ValueError("step expects input_ids with shape [batch, 1]")
        if past_key_values is None:
            past_key_values = [None] * self.config.n_layers
        x = self._embed(input_ids, position_offset=position)
        layer_states = [x]
        next_key_values: List[KeyValue] = []
        for layer, past in zip(self.layers, past_key_values):
            x, key_value = layer.forward_step(x, past)
            next_key_values.append(key_value)
            layer_states.append(x)
        logits = self.lm_head(self.final_norm(x))
        return logits, tuple(next_key_values), None, tuple(layer_states), None

    def forward_exact(
        self,
        input_ids: torch.Tensor,
        recurrent_mask: Optional[torch.Tensor] = None,
    ) -> DecoderOutput:
        """paddingなし系列の逐次KV-cache評価。Vanillaではparallelと同値。"""
        del recurrent_mask
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        logits_by_step = []
        states_by_layer: List[List[torch.Tensor]] = [
            [] for _ in range(self.config.n_layers + 1)
        ]
        key_values = None
        for position in range(input_ids.shape[1]):
            logits, key_values, _, layer_states, _ = self.step(
                input_ids[:, position : position + 1],
                position,
                key_values,
            )
            logits_by_step.append(logits)
            for layer_index, state in enumerate(layer_states):
                states_by_layer[layer_index].append(state)
        return DecoderOutput(
            logits=torch.cat(logits_by_step, dim=1),
            hidden_states=tuple(torch.cat(states, dim=1) for states in states_by_layer),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
