# clean_directories.py

from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Iterable, Set
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def clean_directories(base_dir: Path, remove_dir_names: Iterable[str]) -> int:
    """
    Remove all subfolders (any depth) whose names match (case-insensitive)
    any item in `remove_dir_names`.

    Parameters
    ----------
    base_dir : Path
        Root directory to clean (taken from cfg.data.base_dir in Hydra).
    remove_dir_names : Iterable[str]
        Folder names to delete (e.g. ['swde', 'gml', 'shp']).

    Returns
    -------
    int
        Number of deleted directories.
    """
    base_dir = Path(base_dir).resolve()
    if not base_dir.exists():
        logger.warning("Base directory does not exist: {}", base_dir)
        return 0

    targets: Set[str] = {n.lower() for n in remove_dir_names}
    logger.info("🧹 Cleaning under {} | targets={}", base_dir, ", ".join(sorted(targets)))

    deleted = 0
    for root, dirs, _ in os.walk(base_dir, topdown=True):
        for d in list(dirs):  # iterate over a copy
            if d.lower() in targets:
                dir_path = Path(root) / d
                try:
                    shutil.rmtree(dir_path)
                    dirs.remove(d)
                    deleted += 1
                    logger.info("🗑️  Deleted directory: {}", dir_path)
                except Exception as e:
                    logger.exception("⚠️  Failed to delete {}: {}", dir_path, e)

    if deleted:
        logger.success("✅ Cleanup complete. Deleted {} directories under {}", deleted, base_dir)
    else:
        logger.info("✨ Nothing to delete under {}", base_dir)

    return deleted
