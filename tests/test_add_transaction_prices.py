import duckdb
from omegaconf import OmegaConf

from src.features.add_transaction_prices import run_load_transactions

# NOTE: the db file is deliberately NOT named "egib.duckdb" here. DuckDB names the default
# catalog after the file stem, so an "egib.duckdb" file yields a catalog "egib" that collides
# with the "egib" schema the code creates, making `egib."Transakcje"` ambiguous (BinderException).
# The config default duckdb_path IS artifacts/duckdb/egib.duckdb — see the recorded finding.
_DB = "warehouse.duckdb"


def _make_db(path):
    duckdb.connect(str(path)).close()  # run_load_transactions requires the db file to exist


def _make_tx_parquet(path, rows_sql="SELECT 'p1' AS id, DATE '2021-05-01' AS Data, 100000 AS cena"):
    con = duckdb.connect(":memory:")
    con.execute(f"COPY ({rows_sql}) TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close()


def _cfg(db_path, tx_path, **over):
    base = {
        "features": {
            "enabled": True,
            "add_transaction_prices": {
                "enabled": True,
                "date_column": "Data",
                "write_mode": "replace",
            },
        },
        "data": {"duckdb_path": str(db_path), "transactions_path": str(tx_path)},
    }
    cfg = OmegaConf.create(base)
    return OmegaConf.merge(cfg, OmegaConf.create(over)) if over else cfg


def _table_exists(db_path) -> bool:
    con = duckdb.connect(str(db_path))
    try:
        return (
            con.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema='egib' AND table_name='Transakcje'"
            ).fetchone()
            is not None
        )
    finally:
        con.close()


class TestRunLoadTransactions:
    def test_disabled_does_nothing(self, tmp_path):
        db, tx = tmp_path / _DB, tmp_path / "tx.parquet"
        _make_db(db)
        _make_tx_parquet(tx)
        cfg = _cfg(db, tx, features={"add_transaction_prices": {"enabled": False}})

        run_load_transactions(cfg)

        assert not _table_exists(db)

    def test_missing_db_returns_without_error(self, tmp_path):
        # No db file created — run must log and return, not raise.
        tx = tmp_path / "tx.parquet"
        _make_tx_parquet(tx)
        run_load_transactions(_cfg(tmp_path / "absent.duckdb", tx))

    def test_loads_and_extracts_tx_year(self, tmp_path):
        db, tx = tmp_path / _DB, tmp_path / "tx.parquet"
        _make_db(db)
        _make_tx_parquet(tx)

        run_load_transactions(_cfg(db, tx))

        con = duckdb.connect(str(db))
        try:
            year, n = con.execute(
                'SELECT tx_year, COUNT(*) FROM egib."Transakcje" GROUP BY tx_year'
            ).fetchone()
        finally:
            con.close()
        assert year == 2021
        assert n == 1

    def test_write_mode_skip_keeps_existing_table(self, tmp_path):
        db, tx = tmp_path / _DB, tmp_path / "tx.parquet"
        _make_db(db)
        _make_tx_parquet(tx)
        run_load_transactions(_cfg(db, tx))  # first load

        # Re-run in skip mode with a different parquet — table must stay the original 1-row one.
        _make_tx_parquet(
            tx,
            rows_sql="SELECT 'p2' AS id, DATE '2022-01-01' AS Data, 5 AS cena UNION ALL SELECT 'p3', DATE '2022-02-01', 6",
        )
        cfg = _cfg(db, tx, features={"add_transaction_prices": {"write_mode": "skip"}})
        run_load_transactions(cfg)

        con = duckdb.connect(str(db))
        try:
            n = con.execute('SELECT COUNT(*) FROM egib."Transakcje"').fetchone()[0]
        finally:
            con.close()
        assert n == 1  # unchanged
