import geopandas as gpd
import numpy as np
import pytest
from omegaconf import OmegaConf
from shapely.geometry import Polygon

from src.features.add_uzg_hexs import _get_hex_params, kug_hex_shares


def _rect(x0, y0, x1, y1) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _hex_gdf() -> gpd.GeoDataFrame:
    # One 10x10 hex (area 100 m^2 in EPSG:2180).
    return gpd.GeoDataFrame(
        {"hex_id": ["h1"], "hex_area_m2": [100.0]},
        geometry=[_rect(0, 0, 10, 10)],
        crs="EPSG:2180",
    )


class TestGetHexParams:
    def test_returns_table_and_defaults(self):
        cfg = OmegaConf.create({"pipeline": {"hex": {"table": "hex.Hexagons"}}})
        table, id_col, geom_col = _get_hex_params(cfg)
        assert table == "hex.Hexagons"
        assert id_col == "hex_id"
        assert geom_col == "geometry"

    def test_missing_table_raises(self):
        cfg = OmegaConf.create({"pipeline": {"hex": {}}})
        with pytest.raises(ValueError):
            _get_hex_params(cfg)


class TestKugHexShares:
    def test_two_classes_split_hex_and_area_weighted_bon_mean(self):
        # Bottom half of the hex is class R (bon 3), top half is Ł (bon 5). Each covers 50 of
        # the hex's 100 m^2 → share 0.5 each; area-weighted bon = (3*50 + 5*50)/100 = 4.0.
        kug = gpd.GeoDataFrame(
            {
                "year": [2020, 2020],
                "uzg_ozu_simple": ["R", "Ł"],
                "uzg_bon_score": [3.0, 5.0],
            },
            geometry=[_rect(0, 0, 10, 5), _rect(0, 5, 10, 10)],
            crs="EPSG:2180",
        )

        out = kug_hex_shares(kug, _hex_gdf(), classes=["R", "Ł"]).set_index("hex_id")

        assert out.loc["h1", "uzg_R_share"] == pytest.approx(0.5)
        assert out.loc["h1", "uzg_Ł_share"] == pytest.approx(0.5)
        assert out.loc["h1", "uzg_bon_score_mean"] == pytest.approx(4.0)
        assert out.loc["h1", "year"] == 2020

    def test_partial_coverage_share_below_one(self):
        kug = gpd.GeoDataFrame(
            {"year": [2020], "uzg_ozu_simple": ["R"], "uzg_bon_score": [3.0]},
            geometry=[_rect(0, 0, 10, 2)],  # 20 of 100
            crs="EPSG:2180",
        )

        out = kug_hex_shares(kug, _hex_gdf(), classes=["R"]).set_index("hex_id")

        assert out.loc["h1", "uzg_R_share"] == pytest.approx(0.2)

    def test_fixed_classes_yield_stable_zero_columns(self):
        # A class present in `classes` but absent from the data must still appear as a 0.0 column,
        # so the output schema is stable across units/years regardless of what's on the ground.
        kug = gpd.GeoDataFrame(
            {"year": [2020], "uzg_ozu_simple": ["R"], "uzg_bon_score": [3.0]},
            geometry=[_rect(0, 0, 10, 5)],
            crs="EPSG:2180",
        )

        out = kug_hex_shares(kug, _hex_gdf(), classes=["R", "Ł", "N"]).set_index("hex_id")

        assert out.loc["h1", "uzg_Ł_share"] == 0.0
        assert out.loc["h1", "uzg_N_share"] == 0.0

    def test_no_overlap_returns_empty_frame(self):
        kug = gpd.GeoDataFrame(
            {"year": [2020], "uzg_ozu_simple": ["R"], "uzg_bon_score": [3.0]},
            geometry=[_rect(100, 100, 110, 110)],  # nowhere near the hex
            crs="EPSG:2180",
        )

        out = kug_hex_shares(kug, _hex_gdf(), classes=["R"])

        assert out.empty

    def test_missing_required_column_raises(self):
        kug = gpd.GeoDataFrame(
            {"uzg_ozu_simple": ["R"]},  # no 'year'
            geometry=[_rect(0, 0, 10, 5)],
            crs="EPSG:2180",
        )
        with pytest.raises(ValueError):
            kug_hex_shares(kug, _hex_gdf(), classes=["R"])

    def test_all_nan_bonitacja_leaves_mean_missing(self):
        kug = gpd.GeoDataFrame(
            {"year": [2020], "uzg_ozu_simple": ["R"], "uzg_bon_score": [np.nan]},
            geometry=[_rect(0, 0, 10, 5)],
            crs="EPSG:2180",
        )

        out = kug_hex_shares(kug, _hex_gdf(), classes=["R"]).set_index("hex_id")

        assert out.loc["h1", "uzg_bon_score_mean"] != out.loc["h1", "uzg_bon_score_mean"]  # NaN
