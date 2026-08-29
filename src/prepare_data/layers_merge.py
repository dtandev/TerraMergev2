# src/prepare_data/layers_merge.py
from __future__ import annotations

import re
import shutil
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf

# Some deliveries (confirmed real: rok_2026 "Urządzeniowo-rolne" dataset, ~10 of 21 powiats)
# export every GDB layer under a third naming convention distinct from both the standard
# "EGB_<Target>" and legacy "G5G_<code>" schemes: a long, per-gmina-specific prefix followed by
# "_egb_<target, lowercase, no separators>", optionally with a trailing geometry-type suffix from
# a GIS export tool (e.g. "pow_bartoszycki_m_Bartoszyce_egb_dzialkaewidencyjna",
# "pow_braniewski_M_Braniewo_egb_obiekttrwalezwiazanyzbudynkiem_MultiPolygon"). The per-gmina
# prefix varies by delivery, so it can't be listed as a static alias like the other two schemes --
# matched instead by the "_egb_<canonical>" suffix, case-insensitive, after stripping a trailing
# geometry-type marker.
_GEOMETRY_TYPE_SUFFIXES = (
    "_multipolygon",
    "_multilinestring",
    "_multipoint",
    "_polygon",
    "_linestring",
    "_point",
)

# GDAL appends a trailing "_<n>" disambiguator when an export tool would otherwise write two
# layers under the same name into one dataset -- confirmed real (rok_2026 elbląski delivery:
# "elblski_egb_konturuzytkugruntowego_3", "elblski_egb_konturklasyfikacyjny_3"). Not a meaningful
# part of the target name, so it's stripped the same way a geometry-type suffix is.
_NUMERIC_DISAMBIGUATOR = re.compile(r"_\d+$")


def _matches_egb_suffix(dirname: str, target: str) -> bool:
    """True if `dirname` ends with "_egb_<target>" (case-insensitive, geometry/numeric suffix
    stripped)."""
    lowered = dirname.lower()
    marker = "_egb_"
    idx = lowered.rfind(marker)
    if idx == -1:
        return False
    suffix = lowered[idx + len(marker) :]
    for geom_suffix in _GEOMETRY_TYPE_SUFFIXES:
        if suffix.endswith(geom_suffix):
            suffix = suffix[: -len(geom_suffix)]
            break
    suffix = _NUMERIC_DISAMBIGUATOR.sub("", suffix)
    return suffix == target.lower()


# A separate, unrelated naming quirk: some standard "EGB_<Target>" exports carry a trailing
# 4-digit year (confirmed real: olecki rok_2025 delivery, folder
# "EGB_KonturUzytkuGruntowego_2024" for a unit whose canonical alias is "EGB_KonturUzytkuGruntowego"
# with no year suffix at all). Not part of the alias -- stripped before the exact-alias comparison.
_YEAR_SUFFIX = re.compile(r"_\d{4}$")


def _matches_year_suffixed_alias(dirname: str, aliases: list[str]) -> bool:
    """True if `dirname` ends with "_YYYY" and, with that suffix stripped, exactly matches
    (case-insensitive) one of `aliases`.

    Requires an actual year suffix to strip -- on a case-insensitive filesystem (default on
    macOS), a plain alias name with no year suffix is already found by the direct
    `unit_dir / alias` lookup; matching it again here would add a second Path object for the
    SAME physical directory (case-sensitive Python string equality doesn't dedupe it against
    the first), and `run_layers_merge` would then try to move its contents twice -- the second
    attempt crashing with FileNotFoundError once the first has already emptied/removed it
    (confirmed real crash on real data).
    """
    if not _YEAR_SUFFIX.search(dirname):
        return False
    stripped = _YEAR_SUFFIX.sub("", dirname).lower()
    return any(stripped == alias.lower() for alias in aliases)


def _find_source_dirs(unit_dir: Path, target: str, aliases: list[str]) -> list[Path]:
    """Exact-alias matches (as configured), plus any "_egb_<target>"-suffixed directories, plus
    any exact alias with a trailing "_YYYY" year suffix."""
    found: list[Path] = []
    seen: set[Path] = set()
    for alias in aliases:
        p = unit_dir / alias
        if p.exists() and p not in seen:
            found.append(p)
            seen.add(p)
    for child in unit_dir.iterdir():
        if not child.is_dir() or child in seen:
            continue
        if _matches_egb_suffix(child.name, target) or _matches_year_suffixed_alias(
            child.name, aliases
        ):
            found.append(child)
            seen.add(child)
    return found


def run_layers_merge(cfg: DictConfig) -> None:
    """
    Scala katalogi warstw wg mapy:
      target_layer: [source_layer_1, source_layer_2, ...]
    Oczekiwany layout wejściowy po kroku 'prepare_data':
      base_dir / prepare.output_subdir / {unit_code} / {layer} / year=YYYY / *.parquet

    Działanie:
    - dla każdej jednostki (unit_code) i target_layer:
      - przenieś pliki z {source_layer}/year=YYYY/* → {target_layer}/year=YYYY/*
        (źródło rozpoznane po jawnym aliasie z configu LUB po sufiksie "_egb_<target>")
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

            for source_path in _find_source_dirs(unit_dir, target, sources):
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
