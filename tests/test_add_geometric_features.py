import geopandas as gpd
import pandas as pd
from omegaconf import OmegaConf
from shapely.geometry import Polygon

from src.features.add_geometric_features import (
    _available_feature_makers,
    _cfg_get,
    _compute_features,
    _fmt_seconds,
    _round_numeric_features,
    _run_maker,
    _to_bytes,
)


def _square(s=10.0) -> Polygon:
    return Polygon([(0, 0), (s, 0), (s, s), (0, s)])


class TestCfgGet:
    def test_reads_nested_path(self):
        cfg = OmegaConf.create({"a": {"b": {"c": 7}}})
        assert _cfg_get(cfg, "a.b.c") == 7

    def test_missing_returns_default(self):
        cfg = OmegaConf.create({"a": {}})
        assert _cfg_get(cfg, "a.b.c", default="x") == "x"


class TestToBytes:
    def test_none_and_passthrough(self):
        assert _to_bytes(None) is None
        assert _to_bytes(b"ab") == b"ab"
        assert _to_bytes("ab") == "ab"

    def test_memoryview_and_list(self):
        assert _to_bytes(memoryview(b"hi")) == b"hi"  # via .tobytes()
        assert _to_bytes([104, 105]) == b"hi"  # via bytes(...)


class TestFmtSeconds:
    def test_sub_second_is_ms(self):
        assert _fmt_seconds(0.25).endswith("ms")

    def test_over_second_is_s(self):
        assert _fmt_seconds(2.5).endswith("s") and "ms" not in _fmt_seconds(2.5)


class TestRoundNumericFeatures:
    def test_rounds_floats_but_not_keys_or_strings(self):
        df = pd.DataFrame({"iddzialki": ["a"], "year": [2020], "feat": [1.23456], "name": ["x"]})
        out = _round_numeric_features(df, keys=("iddzialki", "year"), decimals=2)
        assert out["feat"].iloc[0] == 1.23
        assert out["year"].iloc[0] == 2020  # key untouched
        assert out["name"].iloc[0] == "x"  # non-float untouched


class TestAvailableFeatureMakers:
    def test_none_returns_all_seven(self):
        assert len(_available_feature_makers(None)) == 7

    def test_subset_by_name(self):
        makers = _available_feature_makers(["SizeScaleFeatures"])
        assert [m.__name__ for m in makers] == ["SizeScaleFeatures"]

    def test_unknown_name_is_skipped(self):
        assert _available_feature_makers(["NopeFeatures"]) == []


class TestRunMaker:
    def test_failing_maker_returns_empty_frame_indexed_like_input(self):
        gdf = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[_square(), _square()], crs="EPSG:2180")

        class Boom:
            def transform(self, gdf):
                raise RuntimeError("boom")

        out = _run_maker(Boom(), gdf)
        assert out.empty or out.shape[1] == 0
        assert list(out.index) == list(gdf.index)


class TestComputeFeatures:
    def test_builds_feature_matrix_with_keys_and_sorted_columns(self):
        gdf = gpd.GeoDataFrame(
            {"iddzialki": ["a", "b"], "year": [2020, 2020]},
            geometry=[_square(), _square(20)],
            crs="EPSG:2180",
        )

        out = _compute_features(
            gdf,
            id_col="iddzialki",
            year_col="year",
            makers_cfg=["SizeScaleFeatures"],
            mi_samples=10,
            decimals=2,
        )

        assert list(out.columns[:2]) == ["iddzialki", "year"]
        feat_cols = list(out.columns[2:])
        assert feat_cols == sorted(feat_cols)
        assert "area_ha" in feat_cols
        assert out.loc[out["iddzialki"] == "a", "area_ha"].iloc[0] == 0.01  # 10x10 square
        assert all(pd.api.types.is_numeric_dtype(out[c]) for c in feat_cols)
