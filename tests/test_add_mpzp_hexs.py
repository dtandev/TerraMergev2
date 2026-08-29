import geopandas as gpd
import pytest
from omegaconf import OmegaConf
from shapely.geometry import Polygon

from src.features.add_mpzp_hexs import _get_hex_params, mpzp_hex_shares


def _rect(x0, y0, x1, y1) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _hex_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"hex_id": ["h1"], "hex_area_m2": [100.0]},
        geometry=[_rect(0, 0, 10, 10)],
        crs="EPSG:2180",
    )


class TestGetHexParams:
    def test_returns_table_and_defaults(self):
        cfg = OmegaConf.create({"pipeline": {"hex": {"table": "hex.Hexagons"}}})
        assert _get_hex_params(cfg) == ("hex.Hexagons", "hex_id", "geometry")

    def test_missing_table_raises(self):
        with pytest.raises(ValueError):
            _get_hex_params(OmegaConf.create({"pipeline": {"hex": {}}}))


class TestMpzpHexShares:
    def test_two_classes_split_hex(self):
        mpzp = gpd.GeoDataFrame(
            {"year": [2020, 2020], "mpzp_etykieta": ["MN", "U"]},
            geometry=[_rect(0, 0, 10, 5), _rect(0, 5, 10, 10)],
            crs="EPSG:2180",
        )

        out = mpzp_hex_shares(mpzp, _hex_gdf(), classes=["MN", "U"]).set_index("hex_id")

        assert out.loc["h1", "mpzp_MN_share"] == pytest.approx(0.5)
        assert out.loc["h1", "mpzp_U_share"] == pytest.approx(0.5)
        assert out.loc["h1", "year"] == 2020

    def test_partial_coverage_share_below_one(self):
        mpzp = gpd.GeoDataFrame(
            {"year": [2020], "mpzp_etykieta": ["MN"]},
            geometry=[_rect(0, 0, 10, 3)],  # 30 of 100
            crs="EPSG:2180",
        )
        out = mpzp_hex_shares(mpzp, _hex_gdf(), classes=["MN"]).set_index("hex_id")
        assert out.loc["h1", "mpzp_MN_share"] == pytest.approx(0.3)

    def test_fixed_classes_yield_stable_zero_columns(self):
        mpzp = gpd.GeoDataFrame(
            {"year": [2020], "mpzp_etykieta": ["MN"]},
            geometry=[_rect(0, 0, 10, 5)],
            crs="EPSG:2180",
        )
        out = mpzp_hex_shares(mpzp, _hex_gdf(), classes=["MN", "U", "ZP"]).set_index("hex_id")
        assert out.loc["h1", "mpzp_U_share"] == 0.0
        assert out.loc["h1", "mpzp_ZP_share"] == 0.0

    def test_no_overlap_returns_empty_frame(self):
        mpzp = gpd.GeoDataFrame(
            {"year": [2020], "mpzp_etykieta": ["MN"]},
            geometry=[_rect(100, 100, 110, 110)],
            crs="EPSG:2180",
        )
        assert mpzp_hex_shares(mpzp, _hex_gdf(), classes=["MN"]).empty

    def test_missing_required_column_raises(self):
        mpzp = gpd.GeoDataFrame(
            {"mpzp_etykieta": ["MN"]},  # no 'year'
            geometry=[_rect(0, 0, 10, 5)],
            crs="EPSG:2180",
        )
        with pytest.raises(ValueError):
            mpzp_hex_shares(mpzp, _hex_gdf(), classes=["MN"])
