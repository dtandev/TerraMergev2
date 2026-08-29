import geopandas as gpd
import pandas as pd
import pytest
from omegaconf import OmegaConf
from shapely.geometry import Polygon

from src.features.make_empty_hexs import _build_hexes_for_year_v3, _get_hex_node


def _parcels_4326(year=2020, cx=21.0, cy=52.0, s=0.05) -> gpd.GeoDataFrame:
    poly = Polygon([(cx, cy), (cx + s, cy), (cx + s, cy + s), (cx, cy + s)])
    return gpd.GeoDataFrame({"year": [year]}, geometry=[poly], crs="EPSG:4326")


class TestGetHexNode:
    def test_prefers_pipeline_make_hexagons(self):
        cfg = OmegaConf.create(
            {"pipeline": {"make_hexagons": {"year": 2020}}, "make_hexagons": {"year": 1999}}
        )
        assert _get_hex_node(cfg)["year"] == 2020

    def test_falls_back_to_top_level(self):
        cfg = OmegaConf.create({"make_hexagons": {"year": 1999}})
        assert _get_hex_node(cfg)["year"] == 1999

    def test_none_when_absent(self):
        assert _get_hex_node(OmegaConf.create({})) is None


class TestBuildHexesForYear:
    def test_wrong_crs_raises(self):
        gdf = _parcels_4326().to_crs("EPSG:2180")
        with pytest.raises(ValueError):
            _build_hexes_for_year_v3(gdf, year=2020, res=7)

    def test_missing_year_column_raises(self):
        gdf = _parcels_4326().drop(columns=["year"])
        with pytest.raises(KeyError):
            _build_hexes_for_year_v3(gdf, year=2020, res=7)

    def test_no_rows_for_year_raises(self):
        with pytest.raises(ValueError):
            _build_hexes_for_year_v3(_parcels_4326(year=2020), year=1900, res=7)

    def test_produces_hex_grid_for_selected_year(self):
        out = _build_hexes_for_year_v3(_parcels_4326(), year=2020, res=7)

        assert list(out.columns) == ["hex_id", "res", "geometry"]
        assert out.crs.to_epsg() == 4326
        assert len(out) > 0
        assert (out["res"] == 7).all()
        assert out["hex_id"].is_unique
        assert out.geometry.is_valid.all()

    def test_only_the_requested_year_is_dissolved(self):
        # A far-away 2019 parcel must not contribute cells when building 2020.
        near = _parcels_4326(year=2020, cx=21.0, cy=52.0)
        far = _parcels_4326(year=2019, cx=15.0, cy=50.0)
        both = gpd.GeoDataFrame(pd.concat([near, far], ignore_index=True), crs="EPSG:4326")

        out = _build_hexes_for_year_v3(both, year=2020, res=7)
        only_2020 = _build_hexes_for_year_v3(near, year=2020, res=7)

        assert set(out["hex_id"]) == set(only_2020["hex_id"])
