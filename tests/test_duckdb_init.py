import duckdb
import pytest
from omegaconf import OmegaConf

from src.prepare_data.duckdb_init import _load_extension, run_duckdb_init


class _OfflineCon:
    """Stand-in for a DuckDB connection where every LOAD/INSTALL fails — simulates being
    offline with no cached extension copy. (A real DuckDBPyConnection is a C object whose
    `execute` can't be monkeypatched.)"""

    def execute(self, *_a, **_k):
        raise duckdb.IOException("simulated offline")


class TestLoadExtension:
    def test_loads_available_extension(self):
        con = duckdb.connect(":memory:")
        try:
            assert _load_extension(con, "spatial", required=True) is True
        finally:
            con.close()

    def test_optional_extension_failure_is_swallowed(self):
        # An optional extension (httpfs) must degrade to a warning and let init continue.
        assert _load_extension(_OfflineCon(), "httpfs", required=False) is False

    def test_required_extension_failure_reraises(self):
        with pytest.raises(duckdb.IOException):
            _load_extension(_OfflineCon(), "spatial", required=True)


class TestRunDuckdbInit:
    def test_creates_db_file_and_schema(self, tmp_path):
        db_path = tmp_path / "egib.duckdb"
        cfg = OmegaConf.create(
            {"data": {"duckdb_path": str(db_path)}, "duckdb": {"schema": "egib"}}
        )

        run_duckdb_init(cfg)

        assert db_path.exists()
        con = duckdb.connect(str(db_path))
        try:
            schemas = [
                r[0]
                for r in con.execute(
                    "SELECT schema_name FROM information_schema.schemata"
                ).fetchall()
            ]
            assert "egib" in schemas
        finally:
            con.close()

    def test_creates_missing_parent_directories(self, tmp_path):
        db_path = tmp_path / "nested" / "deeper" / "egib.duckdb"
        cfg = OmegaConf.create({"data": {"duckdb_path": str(db_path)}})

        run_duckdb_init(cfg)

        assert db_path.exists()

    def test_applies_threads_and_memory_limit_without_error(self, tmp_path):
        # Smoke: the optional SET threads / PRAGMA memory_limit branches must run cleanly
        # when the config supplies them.
        db_path = tmp_path / "egib.duckdb"
        cfg = OmegaConf.create(
            {
                "data": {"duckdb_path": str(db_path)},
                "duckdb": {"schema": "egib", "threads": 2, "memory_limit": "1GB"},
            }
        )

        run_duckdb_init(cfg)

        assert db_path.exists()
