from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def sel(cfg: DictConfig, path: str, default: Any = None) -> Any:
    """Safe nested selection from a Hydra/OmegaConf config using a dot path."""
    return OmegaConf.select(cfg, path, default=default)
