import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from src.prepare_data.clean_dataset import (
    _cast_year,
    _deduplicate_columns,
    _parse_id,
    _std_geom_name,
)
from src.prepare_data.extract_polygons import _to_uppercase_columns


def _gdf(geom_col_name: str = "geometry") -> gpd.GeoDataFrame:
    df = pd.DataFrame({"a": [1, 2], geom_col_name: [Point(0, 0), Point(1, 1)]})
    gdf = gpd.GeoDataFrame(df, geometry=geom_col_name, crs="EPSG:2180")
    return gdf


class TestCastYear:
    def test_fills_fallback_for_missing_values(self):
        s = pd.Series([2020, None, 2022])
        out = _cast_year(s, fallback=2019)
        assert out.tolist() == [2020, 2019, 2022]
        assert str(out.dtype) == "Int64"

    def test_no_fallback_keeps_na(self):
        s = pd.Series([2020, None])
        out = _cast_year(s, fallback=None)
        assert out.iloc[0] == 2020
        assert pd.isna(out.iloc[1])

    def test_coerces_non_numeric_to_na(self):
        s = pd.Series(["2020", "not_a_year"])
        out = _cast_year(s, fallback=0)
        assert out.tolist() == [2020, 0]


class TestStdGeomName:
    def test_renames_active_geometry_column(self):
        gdf = _gdf(geom_col_name="geom")
        out = _std_geom_name(gdf, target="geometry")
        assert out.geometry.name == "geometry"
        assert "geom" not in out.columns

    def test_noop_when_already_target_name(self):
        gdf = _gdf(geom_col_name="geometry")
        out = _std_geom_name(gdf, target="geometry")
        assert out.geometry.name == "geometry"

    def test_raises_without_active_geometry(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError):
            _std_geom_name(df, target="geometry")


class TestDeduplicateColumns:
    def test_drops_duplicate_named_columns_keeping_first(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])
        out = _deduplicate_columns(df)
        assert list(out.columns) == ["a", "b"]
        assert out["a"].iloc[0] == 1

    def test_noop_when_no_duplicates(self):
        df = pd.DataFrame([[1, 2]], columns=["a", "b"])
        out = _deduplicate_columns(df)
        assert list(out.columns) == ["a", "b"]


class TestParseId:
    PAT = __import__("re").compile(
        r"^\s*(?P<jednostka>\d{6}_\d)\.(?P<obreb>\d{4})\.(?P<nr_dzialki>\d+(?:/\d+)*)\s*$"
    )

    def test_extracts_groups_into_new_columns(self):
        df = pd.DataFrame({"iddzialki": ["281501_2.0003.123/4"]})
        out = _parse_id(df, id_col="iddzialki", pat=self.PAT)
        assert out.loc[0, "jednostka"] == "281501_2"
        assert out.loc[0, "obreb"] == "0003"
        assert out.loc[0, "nr_dzialki"] == "123/4"

    def test_missing_id_column_is_noop(self):
        df = pd.DataFrame({"other": [1]})
        out = _parse_id(df, id_col="iddzialki", pat=self.PAT)
        assert "jednostka" not in out.columns


class TestToUppercaseColumns:
    def test_uppercases_non_geometry_columns(self):
        gdf = gpd.GeoDataFrame(
            {"foo": [1], "bar": [2]},
            geometry=gpd.GeoSeries([Point(0, 0)], name="geometry"),
            crs="EPSG:2180",
        )
        out, rename_map = _to_uppercase_columns(gdf, uppercase_geometry=False)
        assert "FOO" in out.columns and "BAR" in out.columns
        assert out.geometry.name == "geometry"
        assert rename_map == {"foo": "FOO", "bar": "BAR"}

    def test_collision_gets_suffixed(self):
        # 'FOO' already at its own uppercase target; the later 'foo' column, which would
        # collide with it, gets suffixed to 'FOO__2' instead of silently duplicating 'FOO'.
        gdf = gpd.GeoDataFrame(
            {"FOO": [2], "foo": [1]},
            geometry=gpd.GeoSeries([Point(0, 0)], name="geometry"),
            crs="EPSG:2180",
        )
        out, rename_map = _to_uppercase_columns(gdf, uppercase_geometry=False)
        cols = list(out.columns)
        assert cols.count("FOO") == 1
        assert "FOO__2" in cols
        assert rename_map["foo"] == "FOO__2"
