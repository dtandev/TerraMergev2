import geopandas as gpd
import pandas as pd
import pytest
import shapely.wkb
from shapely.geometry import Polygon

from src.features.add_mpzp import (
    _coerce_gdf_geometry,
    _gdf_to_wkb_df,
    _require_columns,
    _sanitize_before_save,
)


def _square(s=1.0) -> Polygon:
    return Polygon([(0, 0), (s, 0), (s, s), (0, s)])


class TestCoerceGdfGeometry:
    def test_accepts_shapely_objects(self):
        df = pd.DataFrame({"id": [1], "geometry": [_square()]})
        out = _coerce_gdf_geometry(df, target_crs="EPSG:2180")
        assert isinstance(out, gpd.GeoDataFrame)
        assert out.crs.to_epsg() == 2180
        assert out.geometry.iloc[0].equals(_square())

    def test_decodes_wkb_bytes(self):
        df = pd.DataFrame({"id": [1], "geometry": [_square().wkb]})
        out = _coerce_gdf_geometry(df)
        assert out.geometry.iloc[0].equals(_square())

    def test_decodes_wkb_hex_string(self):
        df = pd.DataFrame({"id": [1], "geometry": [_square().wkb_hex]})
        out = _coerce_gdf_geometry(df)
        assert out.geometry.iloc[0].equals(_square())

    def test_none_geometry_becomes_missing(self):
        df = pd.DataFrame({"id": [1], "geometry": [None]})
        out = _coerce_gdf_geometry(df)
        assert out.geometry.iloc[0] is None or out.geometry.isna().iloc[0]

    def test_missing_geometry_column_raises(self):
        with pytest.raises(KeyError):
            _coerce_gdf_geometry(pd.DataFrame({"id": [1]}))

    def test_buffer_zero_repairs_self_intersection(self):
        bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
        assert not bowtie.is_valid
        out = _coerce_gdf_geometry(pd.DataFrame({"geometry": [bowtie]}))
        assert out.geometry.iloc[0].is_valid


class TestSanitizeBeforeSave:
    def test_drops_redundant_columns_only(self):
        gdf = gpd.GeoDataFrame(
            {"etykieta": ["A"], "obreb": ["x"], "nr_dzialki": ["1"], "Shape_Area": [1.0]},
            geometry=[_square()],
        )
        out = _sanitize_before_save(gdf)
        assert "etykieta" in out.columns
        assert not ({"obreb", "nr_dzialki", "Shape_Area"} & set(out.columns))

    def test_is_safe_when_columns_absent(self):
        gdf = gpd.GeoDataFrame({"etykieta": ["A"]}, geometry=[_square()])
        out = _sanitize_before_save(gdf)  # must not raise
        assert "etykieta" in out.columns


class TestGdfToWkbDf:
    def test_encodes_geometry_as_wkb_and_drops_geom_col(self):
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[_square()], crs="EPSG:2180")
        df = _gdf_to_wkb_df(gdf)
        assert not isinstance(df, gpd.GeoDataFrame)
        assert "geometry" not in df.columns
        assert "__geom_wkb__" in df.columns
        assert shapely.wkb.loads(df["__geom_wkb__"].iloc[0]).equals(_square())


class TestRequireColumns:
    def test_passes_when_all_present(self):
        gdf = gpd.GeoDataFrame({"a": [1], "b": [2]}, geometry=[_square()])
        _require_columns(gdf, ["a", "b"])  # must not raise

    def test_raises_listing_missing_and_hint(self):
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[_square()])
        with pytest.raises(KeyError, match="ms:plany"):
            _require_columns(gdf, ["a", "data_uchwaly"], layer_hint="ms:plany")
