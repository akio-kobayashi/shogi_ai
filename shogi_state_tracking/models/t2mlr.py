"""T²MLRを小規模将棋decoderへ移植した実装。

参照:
  Cai et al., "T^2MLR: Transformer with Temporal Middle-Layer Recurrence",
  arXiv:2607.15178, especially Eq. (2.2), Eq. (2.4), and Algorithm 1.
  https://arxiv.org/abs/2607.15178

公式Apache-2.0実装:
  https://github.com/princeton-pli/T2MLR
  - src/t2mlr_wrapper/t2mlr_wrapper.py
  - src/t2mlr_wrapper/t2mlr_gate_zoo.py

本ファイルは公式Hugging Face wrapperのコピーではない。今回の固定96状態トークン+
1手1トークンの将棋系列に合わせ、同じbackboneを持つVanillaTransformerへ
middle-layer recurrenceを直接実装した移植版である。主な対応はコード中に注記する。
"""

import math
from typing import List, Optional, Sequence

import torch
from torch import nn

from .config import T2MLRConfig
from .layers import RMSNorm, prepare_sdpa_mask
from .outputs import DecoderOutput
from .transformer import KeyValue, VanillaTransformer


class T2MLRFusion(nn.Module):
    """現在表現xと前tokenの中間表現rを融合するΦ。

    論文Eq. (2.2)および公式`t2mlr_gate_zoo.py`の`gated` mixerに対応する。
    公式実装は複数のgate/projectorを選べるが、本実験では比較要因を減らすため、
    recurrent branchだけを用いる:

        Φ(x, r) = x + tanh(gamma) * sigmoid(W[x;r] + b) * RMSNorm(r)

    gamma=0で初期化するため、学習開始時にはVanillaと厳密に同じ経路になる。
    """

    def __init__(self, d_model: int, eps: float, gate_init: float, rezero_init: float):
        super().__init__()
        self.input_norm = RMSNorm(d_model, eps)
        self.recurrent_norm = RMSNorm(d_model, eps)
        self.gate = nn.Linear(2 * d_model, d_model)
        self.rezero_gamma = nn.Parameter(torch.tensor(float(rezero_init)))
        nn.init.normal_(self.gate.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.gate.bias, math.log(gate_init / (1.0 - gate_init)))

    def forward(
        self, current: torch.Tensor, recurrent: torch.Tensor
    ):
        current_for_gate = self.input_norm(current)
        recurrent_for_mix = self.recurrent_norm(recurrent)
        gate = torch.sigmoid(
            self.gate(torch.cat((current_for_gate, recurrent_for_mix), dim=-1))
        )
        mixed = current + torch.tanh(self.rezero_gamma) * gate * recurrent_for_mix
        return mixed, gate


class T2MLRTransformer(VanillaTransformer):
    """Temporal Middle-Layer Recurrenceを持つcausal decoder。

    学習時は公式`batch_approximate_forward()`と論文Algorithm 1に対応する
    Jacobi型parallel approximationを使う。`exact_recurrence=True`ではKV cacheを
    用い、前tokenのR_tを次tokenのl_startへ逐次注入する。
    """

    config: T2MLRConfig

    def __init__(self, config: T2MLRConfig):
        super().__init__(config)
        self.fusion = T2MLRFusion(
            config.d_model,
            config.norm_eps,
            config.gate_init,
            config.rezero_init,
        )
        self.recurrent_norm = RMSNorm(config.d_model, config.norm_eps)

    @staticmethod
    def _shift_right(states: torch.Tensor) -> torch.Tensor:
        """公式`batch_cache_shift()`に対応。系列境界のcacheを0にする。"""
        zeros = torch.zeros_like(states[:, :1, :])
        return torch.cat((zeros, states[:, :-1, :]), dim=1)

    def _mix_where_active(
        self,
        current: torch.Tensor,
        recurrent: torch.Tensor,
        recurrent_mask: torch.Tensor,
    ):
        mixed, gate = self.fusion(current, recurrent)
        active = recurrent_mask.to(dtype=torch.bool).unsqueeze(-1)
        return torch.where(active, mixed, current), torch.where(
            active, gate, torch.zeros_like(gate)
        )

    def _run_middle(
        self,
        start_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        x = start_states
        states = []
        for layer_index in range(self.config.l_start, self.config.l_end + 1):
            x = self.layers[layer_index](x, attention_mask)
            states.append(x)
        return x, states

    def _update_recurrent(
        self,
        l_end_states: torch.Tensor,
        previous_aligned_cache: Optional[torch.Tensor],
        recurrent_mask: torch.Tensor,
    ) -> torch.Tensor:
        """論文Eq. (2.4): R_t = RMSNorm(h_t^(l_end) + R_(t-1))。

        公式実装の`control_flow`と同様、時間残差はmove区間だけに適用する。
        96状態トークンと<MOVES>まではpromptであり、各tokenのl_end表現をcacheの
        seedにするだけで、前tokenからの再帰融合は行わない。
        """
        seed_state = self.recurrent_norm(l_end_states)
        if self.config.temporal_residual and previous_aligned_cache is not None:
            recurrent_state = self.recurrent_norm(
                l_end_states + previous_aligned_cache
            )
            active = recurrent_mask.to(dtype=torch.bool).unsqueeze(-1)
            return torch.where(active, recurrent_state, seed_state)
        return seed_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        recurrent_mask: Optional[torch.Tensor] = None,
        exact_recurrence: bool = False,
    ) -> DecoderOutput:
        if recurrent_mask is None:
            raise ValueError("T2MLR requires recurrent_mask")
        if recurrent_mask.shape != input_ids.shape:
            raise ValueError("recurrent_mask must have the same shape as input_ids")
        if exact_recurrence:
            if attention_mask is not None and not bool(attention_mask.all()):
                raise ValueError("exact_recurrence currently requires padding-free batches")
            return self.forward_exact(input_ids, recurrent_mask)

        x = self._embed(input_ids)
        attention_mask = prepare_sdpa_mask(attention_mask)
        hidden_states: List[torch.Tensor] = [x]

        # l_startより前はVanillaと同一で、一度だけ計算する。
        for layer_index in range(self.config.l_start):
            x = self.layers[layer_index](x, attention_mask)
            hidden_states.append(x)
        early_states = x

        # Algorithm 1のR<0>: recurrenceなしのmiddle passから初期cacheを得る。
        initial_end, _ = self._run_middle(early_states, attention_mask)
        cache = self._shift_right(self.recurrent_norm(initial_end))

        # 公式batch_approximate_forwardと同じJacobi refinement。
        # 最後のiterationだけ勾配を残す設定も可能だが、既定値0では全て伝播する。
        for iteration in range(self.config.jacobi_depth - 1):
            mixed, _ = self._mix_where_active(early_states, cache, recurrent_mask)
            refined_end, _ = self._run_middle(mixed, attention_mask)
            current_recurrent = self._update_recurrent(
                refined_end, cache, recurrent_mask
            )
            cache = self._shift_right(current_recurrent)
            keep_last = self.config.detach_before_last_n_iterations
            remaining = self.config.jacobi_depth - 1 - iteration
            if keep_last > 0 and remaining > keep_last:
                cache = cache.detach()

        # 最終passだけをLM出力と線形プローブ用hidden_statesに用いる。
        mixed, final_gate = self._mix_where_active(
            early_states, cache, recurrent_mask
        )
        final_end, middle_states = self._run_middle(mixed, attention_mask)
        hidden_states.extend(middle_states)
        recurrent_states = self._update_recurrent(
            final_end, cache, recurrent_mask
        )

        x = final_end
        for layer_index in range(self.config.l_end + 1, self.config.n_layers):
            x = self.layers[layer_index](x, attention_mask)
            hidden_states.append(x)
        logits = self.lm_head(self.final_norm(x))
        return DecoderOutput(
            logits=logits,
            hidden_states=tuple(hidden_states),
            recurrent_states=recurrent_states,
            recurrent_gates=final_gate,
        )

    def step(
        self,
        input_ids: torch.Tensor,
        position: int,
        past_key_values: Optional[Sequence[Optional[KeyValue]]] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        recurrent_active: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ):
        """公式`simple_recurrent_forward()`に対応する厳密な1-token経路。"""
        if input_ids.shape[1] != 1:
            raise ValueError("step expects input_ids with shape [batch, 1]")
        if past_key_values is None:
            past_key_values = [None] * self.config.n_layers
        if recurrent_active is None:
            recurrent_active = torch.ones(
                input_ids.shape[0], dtype=torch.bool, device=input_ids.device
            )

        x = self._embed(input_ids, position_offset=position)
        layer_states = [x]
        next_key_values: List[KeyValue] = []
        gate_value = None
        l_end_state = None

        for layer_index, (layer, past) in enumerate(
            zip(self.layers, past_key_values)
        ):
            if layer_index == self.config.l_start and recurrent_state is not None:
                mixed, gate = self.fusion(x, recurrent_state)
                active = recurrent_active[:, None, None]
                x = torch.where(active, mixed, x)
                gate_value = torch.where(active, gate, torch.zeros_like(gate))
            x, key_value = layer.forward_step(x, past)
            next_key_values.append(key_value)
            layer_states.append(x)
            if layer_index == self.config.l_end:
                l_end_state = x

        if l_end_state is None:
            raise AssertionError("l_end state was not captured")
        seed_state = self.recurrent_norm(l_end_state)
        if recurrent_state is not None and self.config.temporal_residual:
            recurrent_update = self.recurrent_norm(l_end_state + recurrent_state)
            next_recurrent = torch.where(
                recurrent_active[:, None, None], recurrent_update, seed_state
            )
        else:
            next_recurrent = seed_state

        # traceのprefix再生では指定位置以外のvocab projectionを省略できる。
        logits = self.lm_head(self.final_norm(x)) if return_logits else None
        return (
            logits,
            tuple(next_key_values),
            next_recurrent,
            tuple(layer_states),
            gate_value,
        )

    def forward_exact(
        self,
        input_ids: torch.Tensor,
        recurrent_mask: Optional[torch.Tensor] = None,
    ) -> DecoderOutput:
        if recurrent_mask is None or recurrent_mask.shape != input_ids.shape:
            raise ValueError("recurrent_mask must have the same shape as input_ids")
        logits_by_step = []
        recurrent_by_step = []
        gate_by_step = []
        states_by_layer: List[List[torch.Tensor]] = [
            [] for _ in range(self.config.n_layers + 1)
        ]
        key_values = None
        recurrent_state = None

        for position in range(input_ids.shape[1]):
            (
                logits,
                key_values,
                recurrent_state,
                layer_states,
                gate,
            ) = self.step(
                input_ids[:, position : position + 1],
                position,
                key_values,
                recurrent_state,
                recurrent_mask[:, position],
            )
            logits_by_step.append(logits)
            recurrent_by_step.append(recurrent_state)
            if gate is None:
                gate = torch.zeros_like(recurrent_state)
            gate_by_step.append(gate)
            for layer_index, state in enumerate(layer_states):
                states_by_layer[layer_index].append(state)

        return DecoderOutput(
            logits=torch.cat(logits_by_step, dim=1),
            hidden_states=tuple(torch.cat(states, dim=1) for states in states_by_layer),
            recurrent_states=torch.cat(recurrent_by_step, dim=1),
            recurrent_gates=torch.cat(gate_by_step, dim=1),
        )
