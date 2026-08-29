"""Round-trip tests for the DuckDB-based GeoParquet I/O helpers.

These guard two real bugs that only surfaced when the pipeline was actually run end-to-end
(unit tests of the pure logic didn't catch them):

1. write_geoparquet dropped the CRS whenever `crs.to_epsg()` returned None — which is exactly
   the case for the ESRI-flavoured WKT (ESRI:102176 / "ETRS_1989_UWPP_2000_PAS_7") that real
   EGiB GDB/SHP deliveries carry. The parquet then defaulted to CRS84 while the coordinates
   were EPSG:2178, so downstream reprojection produced garbage.
2. Reading those parquets back must go through DuckDB (read_geoparquet), not pyarrow, because
   osgeo/GDAL is imported elsewhere in the same process.
"""

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon

from src.common.duckdb_utils import read_geoparquet, write_geoparquet


def _square(cx=7500000.0, cy=5900000.0, s=100.0) -> Polygon:
    return Polygon([(cx, cy), (cx + s, cy), (cx + s, cy + s), (cx, cy + s)])


class TestGeoParquetRoundTrip:
    def test_epsg_crs_roundtrips(self, tmp_path):
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[_square()], crs="EPSG:2178")
        p = tmp_path / "epsg.parquet"
        write_geoparquet(gdf, p)

        back = read_geoparquet(p)
        assert back.crs is not None and back.crs.equals(CRS.from_epsg(2178))
        assert back.geometry.iloc[0].equals(gdf.geometry.iloc[0])

    def test_esri_wkt_crs_is_preserved_not_dropped(self, tmp_path):
        # ESRI:102176 (CS2000 zone 7) has no clean EPSG code: to_epsg() is None. The old
        # EPSG-only write path silently dropped it -> CRS84. This must NOT happen.
        crs = CRS.from_user_input("ESRI:102176")
        assert crs.to_epsg() is None  # the exact condition that triggered the bug

        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[_square()], crs=crs)
        p = tmp_path / "esri.parquet"
        write_geoparquet(gdf, p)

        back = read_geoparquet(p)
        assert back.crs is not None
        assert back.crs.equals(crs)
        # and it must NOT have degraded to lon/lat CRS84
        assert not back.crs.equals(CRS.from_user_input("OGC:CRS84"))
        # reprojection to PUWG 1992 (2180) yields sensible projected metres, not garbage
        reproj = back.to_crs(2180)
        minx, miny, maxx, maxy = reproj.total_bounds
        assert 100_000 < minx < 900_000 and 100_000 < miny < 900_000

    def test_columns_and_zero_rows(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            {"a": [1, 2], "b": ["x", "y"]},
            geometry=[_square(), _square(cx=7500200.0)],
            crs="EPSG:2178",
        )
        p = tmp_path / "cols.parquet"
        write_geoparquet(gdf, p)
        back = read_geoparquet(p)
        assert set(back.columns) == {"a", "b", "geometry"}
        assert list(back["a"]) == [1, 2]

    def test_read_after_osgeo_import(self, tmp_path):
        # The whole reason read_geoparquet exists: reading must survive osgeo being imported in
        # the process (which breaks gpd.read_parquet). Import it, then read.
        from osgeo import ogr  # noqa: F401

        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[_square()], crs="EPSG:2178")
        p = tmp_path / "after_osgeo.parquet"
        write_geoparquet(gdf, p)
        back = read_geoparquet(p)
        assert len(back) == 1 and back.crs.equals(CRS.from_epsg(2178))
