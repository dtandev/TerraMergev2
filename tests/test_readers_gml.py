import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from src.prepare_data.readers.gml_reader import (
    _flatten_first,
    _flatten_list_columns,
    read_all_layers,
)


def _fixture_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"idDzialki": ["281701_1.0001.1"], "poleEwidencyjne": [123.4]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:2180",
    )


class TestReadAllLayers:
    def test_reads_polygon_layer_from_gml_file(self, tmp_path):
        # geopandas/fiona's GML writer stringifies Python list columns rather than emitting real
        # GML list-type fields (verified empirically), so this fixture only covers scalar columns
        # end-to-end; list-flattening itself is unit-tested directly below against real list values,
        # since that's what GDAL's GML *reader* actually produces for real EGiB deliveries.
        path = tmp_path / "281701_1.gml"
        _fixture_gdf().to_file(path, driver="GML")

        layers = read_all_layers(path)

        assert len(layers) == 1
        gdf = next(iter(layers.values()))
        assert len(gdf) == 1
        assert gdf.iloc[0]["idDzialki"] == "281701_1.0001.1"
        assert gdf.geometry.iloc[0].is_valid

    def test_missing_file_returns_empty(self, tmp_path):
        layers = read_all_layers(tmp_path / "does_not_exist.gml")
        assert layers == {}


class TestFlattenListColumns:
    def test_flatten_first_returns_first_element(self):
        assert _flatten_first(["R", "B"]) == "R"
        assert _flatten_first(("IV",)) == "IV"

    def test_flatten_first_handles_numpy_ndarray(self):
        # Regression guard: fiona/pyogrio return real GML list-type fields as numpy.ndarray, not
        # plain list/tuple (e.g. array(['R', 'Ps', 'Ls'], dtype='<U2')) — an isinstance check that
        # only covers list/tuple silently fails to flatten these, and the leftover ndarray later
        # crashes DuckDB's `register()` with "Data type '<U...' not recognized" (verified against
        # a real GML file — see extract_polygons.py's write path).
        result = _flatten_first(np.array(["R", "Ps", "Ls"]))
        assert result == "R"
        # Indexing a numpy.ndarray yields a numpy scalar (numpy.str_), not a native Python str —
        # DuckDB's register() rejects the numpy scalar ("Unsupported string type: no clue what
        # this string is"), so this must have been unwrapped to a plain str via .item().
        assert type(result) is str

    def test_flatten_first_passes_through_scalars(self):
        assert _flatten_first("IV") == "IV"
        assert _flatten_first(None) is None

    def test_flatten_first_empty_list_returns_none(self):
        assert _flatten_first([]) is None
        assert _flatten_first(np.array([])) is None

    def test_flatten_list_columns_only_touches_known_fields(self):
        gdf = gpd.GeoDataFrame(
            {
                "OZU": [np.array(["R", "B"])],
                "OZK": [["IV"]],
                "idDzialki": ["281701_1.0001.1"],  # not in the flatten list — must stay untouched
            },
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:2180",
        )
        out = _flatten_list_columns(gdf)
        assert out["OZU"].iloc[0] == "R"
        assert out["OZK"].iloc[0] == "IV"
        assert out["idDzialki"].iloc[0] == "281701_1.0001.1"
        assert not isinstance(out["OZU"].iloc[0], np.ndarray)
