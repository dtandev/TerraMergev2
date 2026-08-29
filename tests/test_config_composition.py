"""
Regression guard for the config-nesting bug class described in audit.md: several conf/*.yaml
groups used to declare `# @package _global_`, which silently flattened their keys to the config
root while most of the Python code read them as nested under their group name (`pipeline.*`,
`features.*`, ...). OmegaConf.select()'s `default=` swallowed the mismatch instead of raising, so
whole pipeline steps were always-skipped or always-crashed without any error at startup.

This test composes the real conf/ tree with hydra.compose() and asserts the exact nested paths
the code relies on actually resolve — so a future re-introduction of `@package _global_` (or a
typo'd key) fails a test instead of failing silently in production.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONF_DIR = str((Path(__file__).parent.parent / "conf").resolve())

_MISSING = object()


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("TERRAMERGE_BASE_DIR", "/tmp/fake_egib")
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        yield compose(config_name="config")


NESTED_KEYS = [
    "prepare.enabled",
    "prepare.clean.enabled",
    "prepare.swde_crs",
    "features.enabled",
    "features.add_uzg.enabled",
    "features.add_mpzp.enabled",
    "features.add_geometric_features.add_to_duckdb.enabled",
    "pipeline.make",
    "pipeline.make_hexagons.enabled",
    "pipeline.add_parcels_data.enabled",
    "pipeline.add_transactions_data.enabled",
    "pipeline.add_mpzp_data.enabled",
    "pipeline.add_kug_data.enabled",
    "pipeline.resolution",
    "pipeline.hex.table",
    "pipeline.egib.table",
    "pipeline.layer_defaults.enforce_crs",
    "dataset.enabled",
    "dataset.resolution",
    "dataset.join_y_label.enabled",
    "dataset.calculate_neighborhood.out_table",
    "model.enabled",
    "model.valid_years",
    "model.train_max_year",
    "model.n_perm_repeats",
    "duckdb.schema",
    "duckdb.init",
]


class TestConfigNestingRegression:
    @pytest.mark.parametrize("key", NESTED_KEYS)
    def test_key_resolves_and_is_not_none(self, cfg, key):
        val = OmegaConf.select(cfg, key, default=_MISSING)
        assert val is not _MISSING, f"'{key}' is missing from the composed config"
        assert val is not None, f"'{key}' resolved to None"

    def test_no_stray_root_level_enabled_key(self, cfg):
        # Before the @package _global_ fix, prepare/features/dataset default.yaml all landed an
        # `enabled` key at the config root, clobbering each other. There must be no such key now.
        assert OmegaConf.select(cfg, "enabled", default=_MISSING) is _MISSING

    def test_base_dir_required_env_var_is_enforced(self, monkeypatch):
        monkeypatch.delenv("TERRAMERGE_BASE_DIR", raising=False)
        with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
            bad_cfg = compose(config_name="config")
            with pytest.raises(Exception):
                # data.base_dir has no fallback default -> must fail loudly, not silently None.
                OmegaConf.to_container(bad_cfg, resolve=True)
