# src/prepare_data/layers_merge.py
from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def run_layers_merge(cfg: DictConfig) -> None:
    """
    Scala katalogi warstw wg mapy:
      target_layer: [source_layer_1, source_layer_2, ...]
    Oczekiwany layout wejściowy po kroku 'prepare_data':
      base_dir / prepare.output_subdir / {unit_code} / {layer} / year=YYYY / *.parquet

    Działanie:
    - dla każdej jednostki (unit_code) i target_layer:
      - przenieś pliki z {source_layer}/year=YYYY/* → {target_layer}/year=YYYY/*
      - usuwaj puste katalogi źródłowe
    - kolizje nazw plików:
      - jeśli prepare.merge.overwrite=true → nadpisz
      - w przeciwnym razie pomiń i zaloguj ostrzeżenie
    """
    base_dir = Path(OmegaConf.select(cfg, "data.base_dir")).expanduser().resolve()
    out_subdir = str(OmegaConf.select(cfg, "prepare.output_subdir", default="parquets"))
    overwrite = bool(OmegaConf.select(cfg, "prepare.merge.overwrite", default=False))
    layer_map: dict[str, list[str]] = dict(
        OmegaConf.select(cfg, "prepare.layer_name_map", default={})
    )

    out_root = base_dir / out_subdir
    logger.info("START layers_merge | root={}", out_root)

    if not out_root.exists():
        logger.warning("Katalog wyjściowy nie istnieje (pomijam merge): {}", out_root)
        return

    units = [d for d in out_root.iterdir() if d.is_dir()]
    if not units:
        logger.warning("Brak jednostek (podkatalogów) w: {}", out_root)
        return

    for unit_dir in units:
        for target, sources in layer_map.items():
            target_path = unit_dir / target
            target_path.mkdir(parents=True, exist_ok=True)

            for source in sources:
                source_path = unit_dir / source
                if not source_path.exists():
                    logger.debug("Brak źródła: {}", source_path)
                    continue

                # iterujemy po year=XXXX
                for year_dir in [p for p in source_path.iterdir() if p.is_dir()]:
                    for f in year_dir.iterdir():
                        dest = target_path / year_dir.name / f.name
                        dest.parent.mkdir(parents=True, exist_ok=True)

                        if dest.exists():
                            if overwrite:
                                try:
                                    dest.unlink()
                                except Exception as e:
                                    logger.warning(
                                        "Nie mogę usunąć istniejącego pliku ({}): {}", dest, e
                                    )
                            else:
                                logger.warning("Pominięto (istnieje): {}", dest)
                                continue

                        try:
                            shutil.move(str(f), str(dest))
                            logger.debug("→ {} → {}", f, dest)
                        except Exception as e:
                            logger.exception("Błąd przenoszenia {} → {}: {}", f, dest, e)

                # Spróbuj usunąć puste drzewo źródłowe
                try:
                    shutil.rmtree(source_path, ignore_errors=True)
                except Exception as e:
                    logger.debug("Nie usunięto {}: {}", source_path, e)

    logger.success("LAYERS_MERGE zakończone | root={}", out_root)
