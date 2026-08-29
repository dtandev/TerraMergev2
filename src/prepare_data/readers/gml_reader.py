"""Reader for GML EGiB deliveries.

GML uses the same national EGiB schema as GDB (identical layer names like
`EGB_DzialkaEwidencyjna`, identical field names like `idDzialki`/`OZU`/`OZK`) — confirmed by
comparing a real GML unit against the GDB version of the same unit. GDAL's native GML driver
reads it directly; no custom parsing needed. The one real difference: some fields
(`numerKW`, `OFU`, `powierzchnia`, `powierzchnia_uom`, `OZU`, `OZK`) come back as list-valued
(`StringList`/`RealList`) in GML where GDB has them flattened to a scalar — as `numpy.ndarray`
values (not plain Python list/tuple), confirmed against real data (e.g. a parcel with multiple
land-use zones: `array(['R', 'Ps', 'Ls'], dtype='<U2')`). Downstream code (`src/features/add_uzg.py`)
expects a single scalar classification code per row (matched against a land-classification regex),
so these are flattened to their first element here — matching what the GDB delivery already
provides, not introducing new multi-valued semantics. This does lose information for genuinely
multi-valued parcels (only the first zone survives), but that's the same limitation the existing
GDB pipeline already has, not a new one.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from loguru import logger
from osgeo import ogr

_LIST_COLUMNS_TO_FLATTEN = ("numerKW", "OFU", "powierzchnia", "powierzchnia_uom", "OZU", "OZK")


def _polygon_layer_names(gml_path: Path) -> list[str]:
    ds = ogr.Open(str(gml_path), 0)
    if ds is None:
        logger.error("Nie można otworzyć GML: {}", gml_path)
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


def _flatten_first(value):
    # fiona/pyogrio return GML list-type fields as numpy.ndarray, not a plain list/tuple —
    # verified against real EGiB GML deliveries (a value like array(['R', 'Ps'], dtype='<U2')).
    # numpy arrays also choke DuckDB's `register()` ("Data type '<U...' not recognized") if left
    # unflattened, so this must catch them, not just list/tuple.
    if isinstance(value, (list, tuple, np.ndarray)):
        if not len(value):
            return None
        first = value[0]
        # Indexing a numpy.ndarray yields a numpy scalar (numpy.str_/numpy.float64, ...), not a
        # native Python type — DuckDB's register() rejects those ("Unsupported string type: no
        # clue what this string is"), confirmed against a real GML `numerKW`/`OFU` column.
        return first.item() if isinstance(first, np.generic) else first
    return value


def _flatten_list_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for col in _LIST_COLUMNS_TO_FLATTEN:
        if (
            col in gdf.columns
            and gdf[col].map(lambda v: isinstance(v, (list, tuple, np.ndarray))).any()
        ):
            gdf[col] = gdf[col].map(_flatten_first)
    return gdf


def read_all_layers(gml_path: Path) -> dict[str, gpd.GeoDataFrame]:
    """Read every polygon layer from a .gml file into {layer_name: GeoDataFrame}."""
    layers: dict[str, gpd.GeoDataFrame] = {}
    for name in _polygon_layer_names(gml_path):
        try:
            gdf = gpd.read_file(str(gml_path), layer=name)
            gdf = _flatten_list_columns(gdf)
            layers[name] = gdf
        except Exception:
            logger.exception("Błąd odczytu warstwy {} z {}", name, gml_path.name)
    return layers


__all__ = ["read_all_layers"]
