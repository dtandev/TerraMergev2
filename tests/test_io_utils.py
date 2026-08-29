import sys

import pytest
from loguru import logger

from src.common.io_utils import _ensure_dir, setup_logging


@pytest.fixture(autouse=True)
def restore_logger():
    """setup_logging() mutates the process-wide loguru logger (removes all sinks, adds its
    own file/console sinks with enqueue=True). Snapshot nothing, but after each test reset to
    a single stderr sink so an enqueued file sink never leaks into unrelated tests."""
    yield
    logger.remove()
    logger.add(sys.stderr)


class TestEnsureDir:
    def test_creates_nested_dirs(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        _ensure_dir(target)
        assert target.is_dir()

    def test_is_idempotent(self, tmp_path):
        _ensure_dir(tmp_path / "x")
        _ensure_dir(tmp_path / "x")  # must not raise
        assert (tmp_path / "x").is_dir()


class TestSetupLogging:
    def test_creates_log_dir_and_returns_log_file_path(self, tmp_path):
        log_dir = tmp_path / "logs"  # does not exist yet
        log_file = setup_logging(log_dir, "myrun")

        assert log_dir.is_dir()
        assert log_file == log_dir.resolve() / "myrun.log"

    def test_writes_messages_to_the_log_file(self, tmp_path):
        log_file = setup_logging(tmp_path, "run")
        logger.info("hello-marker")
        logger.remove()  # flush + join the enqueued file sink before reading

        assert log_file.exists()
        assert "hello-marker" in log_file.read_text()

    def test_file_level_filters_out_lower_levels(self, tmp_path):
        log_file = setup_logging(tmp_path, "run", file_level="WARNING")
        logger.info("info-should-be-filtered")
        logger.warning("warning-should-appear")
        logger.remove()

        contents = log_file.read_text()
        assert "warning-should-appear" in contents
        assert "info-should-be-filtered" not in contents
