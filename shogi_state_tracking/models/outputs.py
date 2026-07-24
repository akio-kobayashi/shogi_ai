from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DecoderOutput:
    """学習出力と線形プローブ用中間表現。"""

    logits: torch.Tensor
    # embedding出力 + 各Transformer block出力（合計n_layers + 1）
    hidden_states: Tuple[torch.Tensor, ...]
    # T²MLRのみ。各tokenを処理した後のR_t。VanillaではNone。
    recurrent_states: Optional[torch.Tensor] = None
    # T²MLRの最終Jacobi passにおける再帰gate。
    recurrent_gates: Optional[torch.Tensor] = None
