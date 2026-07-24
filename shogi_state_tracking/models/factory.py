from typing import Union

from .config import ModelConfig, T2MLRConfig
from .t2mlr import T2MLRTransformer
from .transformer import VanillaTransformer


def build_model(
    model_type: str,
    config: Union[ModelConfig, T2MLRConfig],
):
    key = model_type.strip().lower()
    if key == "vanilla":
        return VanillaTransformer(config)
    if key in {"t2mlr", "t^2mlr", "t²mlr"}:
        if not isinstance(config, T2MLRConfig):
            config = T2MLRConfig(**config.to_dict())
        return T2MLRTransformer(config)
    raise ValueError("unknown model_type: {}".format(model_type))
