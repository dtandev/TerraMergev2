"""Reader for legacy ESRI Shapefile EGiB deliveries.

Unlike GDB/GML (one container = many layers), a SHP delivery is one folder per unit containing
many individually-named `.shp` files, one per thematic layer, using cryptic legacy codes
(`G5G_DZE.shp` = parcels, `G5G_BUD.shp` = buildings, ...) instead of the modern `EGB_*` layer
names. `conf/prepare/default.yaml`'s `prepare.layer_name_map` already lists these G5 aliases per
canonical layer name (used today by `run_layers_merge` to merge already-extracted Parquet
layers) — this reader reuses the same map to identify which `.shp` file is which layer, so the
alias list only needs to be maintained in one place.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
from loguru import logger

# Legacy `.prj` sidecars for these deliveries carry ESRI-flavored WKT (name like
# "ETRS_1989_UWPP_2000_PAS_7") that pyproj's `CRS.to_epsg()` cannot resolve to a clean EPSG code
# (returns None) — confirmed against a real `.prj` file. Lowering `to_epsg(min_confidence=...)`
# is NOT a safe fix: at min_confidence=20 pyproj matched this exact WKT to EPSG:6870 (Albania TM
# 2010) instead of the correct Polish CS2000 zone. Since Polish EGiB deliveries only ever use one
# of the four CS2000 zones (5/6/7/8 -> central meridian 15/18/21/24), and the zone number is
# spelled out verbatim as "PAS_<n>" in this WKT's name, map it explicitly instead of guessing.
_PAS_TO_EPSG = {5: 2176, 6: 2177, 7: 2178, 8: 2179}
_PAS_PATTERN = re.compile(r"PAS[_ ]?(\d)", re.IGNORECASE)


def _resolve_unresolvable_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fix legacy ESRI `.prj` CRS that pyproj can't auto-resolve to an EPSG code (see above)."""
    crs = gdf.crs
    if crs is None or crs.to_epsg() is not None:
        return gdf

    match = _PAS_PATTERN.search(crs.name or "")
    if match is None:
        logger.warning("Nie rozpoznano EPSG dla CRS pliku SHP: {}", crs.name)
        return gdf

    epsg = _PAS_TO_EPSG.get(int(match.group(1)))
    if epsg is None:
        logger.warning("Nieznana strefa PAS_{} w CRS: {}", match.group(1), crs.name)
        return gdf

    return gdf.set_crs(f"EPSG:{epsg}", allow_override=True)


def _alias_to_canonical(layer_name_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """Invert {canonical: [aliases]} into {alias_upper: [canonical, ...]} for case-insensitive
    lookup. An alias can legitimately feed more than one canonical layer — confirmed on real data:
    `G5G_KKL.shp` carries both a land-use attribute (`G5OZU`) and a soil-classification attribute
    (`G5OZK`) on the very same geometry, so it is the real source for both
    `KonturUzytkuGruntowego` and `KonturKlasyfikacyjny` (the legacy G5 system never split them into
    two files the way the modern EGB schema splits them into two feature classes)."""
    out: dict[str, list[str]] = {}
    for canonical, aliases in layer_name_map.items():
        for alias in aliases:
            out.setdefault(alias.upper(), []).append(canonical)
    return out


def read_all_layers(
    shp_paths: Iterable[Path], layer_name_map: dict[str, list[str]]
) -> dict[str, gpd.GeoDataFrame]:
    """
    Read every recognized `.shp` file (belonging to one unit) into {canonical_layer: GeoDataFrame}.

    Takes an explicit list of `.shp` paths rather than a directory — real deliveries don't always
    keep one unit's shapefiles in a single flat folder (see `_discover_units` in
    `extract_polygons.py`), so the caller does the discovery/grouping and this function just reads.
    `.shp` files whose basename doesn't match any alias in `layer_name_map` are skipped (logged at
    debug level) — they're typically non-polygon or auxiliary layers this pipeline doesn't consume.
    A file matching more than one canonical layer (see `_alias_to_canonical`) is read once and its
    GeoDataFrame assigned to every matching canonical key.
    """
    alias_map = _alias_to_canonical(layer_name_map)
    layers: dict[str, gpd.GeoDataFrame] = {}

    for shp_path in sorted(shp_paths):
        canonicals = alias_map.get(shp_path.stem.upper())
        if not canonicals:
            logger.debug("Pomijam nierozpoznaną warstwę SHP: {}", shp_path.name)
            continue
        try:
            gdf = gpd.read_file(str(shp_path))
        except Exception:
            logger.exception("Błąd odczytu {}", shp_path)
            continue

        gdf = _resolve_unresolvable_crs(gdf)

        geom_types = set(gdf.geom_type.dropna().unique().tolist())
        if not geom_types & {"Polygon", "MultiPolygon"}:
            logger.debug("Pomijam {} ({}) — nie jest warstwą poligonową", shp_path.name, geom_types)
            continue

        for canonical in canonicals:
            if canonical in layers:
                logger.warning(
                    "Więcej niż jeden plik SHP mapuje się na warstwę '{}' — nadpisuję poprzedni ({}).",
                    canonical,
                    shp_path.name,
                )
            layers[canonical] = gdf.copy() if len(canonicals) > 1 else gdf

    return layers


__all__ = ["read_all_layers"]
