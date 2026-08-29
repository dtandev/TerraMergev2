import geopandas as gpd
from omegaconf import OmegaConf
from shapely.geometry import Point

from src.common.config_utils import sel
from src.common.duckdb_utils import (
    _detect_srid,
    connect_duckdb,
    save_geodf_as_ewkb_geometry,
    write_geoparquet,
)


class TestSel:
    def test_present_nested_key(self):
        cfg = OmegaConf.create({"a": {"b": {"c": 42}}})
        assert sel(cfg, "a.b.c") == 42

    def test_missing_key_returns_default(self):
        cfg = OmegaConf.create({"a": {"b": 1}})
        assert sel(cfg, "a.b.c", default="fallback") == "fallback"

    def test_typo_d_key_returns_default_not_raise(self):
        cfg = OmegaConf.create({"dataset": {"join_y_label": {"enabled": True}}})
        assert sel(cfg, "dataset.join_y_label.enable", default=True) == True  # noqa: E712
        assert sel(cfg, "dataset.join_y_label.enabled", default=False) == True  # noqa: E712


class TestDuckdbUtils:
    def test_roundtrip_write_and_read_back(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        gdf = gpd.GeoDataFrame(
            {"hex_id": ["a", "b"], "year": [2020, 2021]},
            geometry=gpd.GeoSeries([Point(0, 0), Point(1, 1)], crs="EPSG:2180"),
        )
        n = save_geodf_as_ewkb_geometry(
            db_path=db_path,
            gdf=gdf,
            table="main.test_table",
            srid=2180,
            casts={"hex_id": "VARCHAR", "year": "INT"},
        )
        assert n == 2

        con = connect_duckdb(db_path)
        try:
            rows = con.execute("SELECT COUNT(*) FROM main.test_table").fetchone()[0]
            assert rows == 2
        finally:
            con.close()

    def test_detect_srid_within_same_connection(self, tmp_path):
        # _detect_srid's CRS metadata is only reliably visible within the connection that wrote
        # it (see the docstring in src/common/duckdb_utils.py) — this test reflects that reality
        # rather than asserting cross-connection detection, which does not work in the currently
        # installed DuckDB spatial extension build.
        db_path = tmp_path / "test.duckdb"
        con = connect_duckdb(db_path)
        try:
            con.execute("CREATE SCHEMA IF NOT EXISTS main;")
            con.execute(
                "CREATE TABLE main.t AS SELECT ST_SetCRS(ST_Point(0, 0), 'EPSG:2180') AS geometry"
            )
            assert _detect_srid(con, "main.t", "geometry") == 2180
        finally:
            con.close()

    def test_detect_srid_across_reconnect_falls_back_to_none(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        gdf = gpd.GeoDataFrame(
            {"hex_id": ["a"]},
            geometry=gpd.GeoSeries([Point(0, 0)], crs="EPSG:2180"),
        )
        save_geodf_as_ewkb_geometry(db_path=db_path, gdf=gdf, table="main.test_table2", srid=2180)

        con = connect_duckdb(db_path)
        try:
            assert _detect_srid(con, "main.test_table2", "geometry") is None
        finally:
            con.close()

    def test_write_geoparquet_handles_zero_row_geodataframe(self, tmp_path):
        # Regression guard: a `.apply()` over an empty GeoSeries keeps the geopandas "geometry"
        # extension dtype (nothing to compute, so it never coerces to plain object/bytes) —
        # confirmed against real EGiB deliveries where a restrictions/limitations layer ("RST",
        # "OZN") legitimately has 0 features for a given unit/year. Without the `.astype(object)`
        # fix, DuckDB's register() rejects that column: "Data type 'geometry' not recognized".
        gdf = gpd.GeoDataFrame(
            {"id": []},
            geometry=gpd.GeoSeries([], dtype="geometry"),
            crs="EPSG:2178",
        )
        out_path = tmp_path / "empty.parquet"

        write_geoparquet(gdf, out_path)

        assert out_path.exists()
        con = connect_duckdb(out_path.parent / "unused.duckdb")
        try:
            con.execute("INSTALL spatial; LOAD spatial;")
            row = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')"
            ).fetchone()
            assert row[0] == 0
        finally:
            con.close()
