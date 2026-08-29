from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Repository root: src/common/config_loader.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONF_DIR = _REPO_ROOT / "conf"


def load_config(
    overrides: Iterable[str] | None = None,
    conf_dir: Path | str = DEFAULT_CONF_DIR,
) -> DictConfig:
    """Compose the pipeline config from ``conf/`` and apply CLI dot-list overrides.

    Replaces the former Hydra composition. ``conf/config.yaml`` lists the per-stage
    group files under an ``includes`` mapping (``group_name: relative/path.yaml``);
    each is loaded and nested under its group key, then the root keys (``data``,
    ``duckdb``, ``wfs``, ``logging``) are merged on top.

    ``overrides`` are Hydra-style ``key.path=value`` strings (e.g. ``sys.argv[1:]``),
    parsed via :func:`OmegaConf.from_dotlist`. Interpolations — ``${oc.env:VAR}`` and
    cross-references like ``${pipeline.resolution}`` — resolve lazily on access, so
    ``load_dotenv()`` must run before the config is read.
    """
    conf_dir = Path(conf_dir)
    root = OmegaConf.load(conf_dir / "config.yaml")

    includes = root.pop("includes", {}) or {}
    cfg = OmegaConf.create()
    for group, rel_path in includes.items():
        cfg[group] = OmegaConf.load(conf_dir / str(rel_path))

    cfg = OmegaConf.merge(cfg, root)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    # The pipeline adds/reads keys not present at compose time; keep the config open.
    OmegaConf.set_struct(cfg, False)
    return cfg  # type: ignore[return-value]
