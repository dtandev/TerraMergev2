import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from src.prepare_data.readers.gdb_reader import _polygon_layer_names, read_all_layers


def _polygon_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"G5IDD": ["281701_1.0001.1"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:2180",
    )


def _point_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:2180",
    )


@pytest.fixture
def gdb_with_mixed_layers(tmp_path):
    """A .gdb holding one polygon layer and one point layer."""
    gdb = tmp_path / "delivery.gdb"
    _polygon_gdf().to_file(gdb, driver="OpenFileGDB", layer="DzialkaEwidencyjna")
    _point_gdf().to_file(gdb, driver="OpenFileGDB", layer="PunktGraniczny", mode="a")
    return gdb


class TestPolygonLayerNames:
    def test_returns_only_polygon_layers(self, gdb_with_mixed_layers):
        names = _polygon_layer_names(gdb_with_mixed_layers)
        assert names == ["DzialkaEwidencyjna"]

    def test_missing_gdb_returns_empty(self, tmp_path):
        assert _polygon_layer_names(tmp_path / "nope.gdb") == []


class TestReadAllLayers:
    def test_reads_polygon_layer_and_skips_points(self, gdb_with_mixed_layers):
        layers = read_all_layers(gdb_with_mixed_layers)

        assert set(layers.keys()) == {"DzialkaEwidencyjna"}
        assert layers["DzialkaEwidencyjna"].iloc[0]["G5IDD"] == "281701_1.0001.1"

    def test_missing_gdb_returns_empty_dict(self, tmp_path):
        assert read_all_layers(tmp_path / "nope.gdb") == {}
