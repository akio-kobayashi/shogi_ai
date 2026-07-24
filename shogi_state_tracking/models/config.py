from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
    """VanillaとT²MLRで共有するbackbone設定。"""

    vocab_size: int
    max_seq_len: int = 640
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers <= 0 or self.d_ff <= 0:
            raise ValueError("n_layers and d_ff must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class T2MLRConfig(ModelConfig):
    """T²MLR固有設定。

    layer indexは0始まりで、l_startとl_endをともに含む。
    8層の既定値3..4は中央2層（25%）を再帰ブロックとする。
    """

    l_start: int = 3
    l_end: int = 4
    jacobi_depth: int = 4
    gate_init: float = 0.2
    rezero_init: float = 0.0
    temporal_residual: bool = True
    detach_before_last_n_iterations: int = 0

    def validate(self) -> None:
        super().validate()
        if not 0 <= self.l_start <= self.l_end < self.n_layers:
            raise ValueError("require 0 <= l_start <= l_end < n_layers")
        if self.jacobi_depth <= 0:
            raise ValueError("jacobi_depth must be positive")
        if not 0.0 < self.gate_init < 1.0:
            raise ValueError("gate_init must be in (0, 1)")
        if self.detach_before_last_n_iterations < 0:
            raise ValueError("detach_before_last_n_iterations must be non-negative")


def parameter_matched_vanilla_config(t2mlr: T2MLRConfig) -> ModelConfig:
    """T²MLRの追加parameterをFFN幅で補ったVanilla設定を返す。

    T2MLRFusionとcache RMSNormの追加数は 2*d_model^2 + 4*d_model + 1。
    Vanillaの全FFN幅を1増やすと 2*d_model*n_layers parameter増える。
    最も近い整数幅を選び、語彙・層数・attention幅などは同一に保つ。
    """
    t2mlr.validate()
    recurrent_parameters = (
        2 * t2mlr.d_model * t2mlr.d_model + 4 * t2mlr.d_model + 1
    )
    per_ff_unit = 2 * t2mlr.d_model * t2mlr.n_layers
    extra_ff = max(1, round(recurrent_parameters / per_ff_unit))
    return ModelConfig(
        vocab_size=t2mlr.vocab_size,
        max_seq_len=t2mlr.max_seq_len,
        d_model=t2mlr.d_model,
        n_layers=t2mlr.n_layers,
        n_heads=t2mlr.n_heads,
        d_ff=t2mlr.d_ff + extra_ff,
        dropout=t2mlr.dropout,
        norm_eps=t2mlr.norm_eps,
        tie_embeddings=t2mlr.tie_embeddings,
    )
