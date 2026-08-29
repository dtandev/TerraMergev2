"""Reader for ESRI File Geodatabase (.gdb) EGiB deliveries."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from loguru import logger
from osgeo import ogr


def _polygon_layer_names(gdb_path: Path) -> list[str]:
    """Return names of Polygon/MultiPolygon layers in a .gdb."""
    ds = ogr.Open(str(gdb_path), 0)
    if ds is None:
        logger.error("Nie można otworzyć GDB: {}", gdb_path)
        return []

    names: list[str] = []
    for i in range(ds.GetLayerCount()):
        layer = ds.GetLayerByIndex(i)
        name = layer.GetName()
        geom_type = ogr.GeometryTypeToName(layer.GetGeomType())
        if geom_type in ("Polygon", "Multi Polygon", "MultiPolygon"):
            names.append(name)
        else:
            logger.debug("Pomijam warstwę nie-poligonową: {} ({})", name, geom_type)
    return names


def read_all_layers(gdb_path: Path) -> dict[str, gpd.GeoDataFrame]:
    """Read every polygon layer from a .gdb into {layer_name: GeoDataFrame}."""
    layers: dict[str, gpd.GeoDataFrame] = {}
    for name in _polygon_layer_names(gdb_path):
        try:
            layers[name] = gpd.read_file(str(gdb_path), layer=name)
        except Exception:
            logger.exception("Błąd odczytu warstwy {} z {}", name, gdb_path.name)
    return layers
