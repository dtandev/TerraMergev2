import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.modeling.train_model import (
    _to_bytes_safe,
    build_X_like,
    fdr_bh,
    load_feature_meta,
    prepare_Xy_for_lgb,
    save_config_snapshot,
    save_feature_meta,
    temporal_holdout_split,
)


class TestToBytesSafe:
    def test_variants(self):
        assert _to_bytes_safe(None) is None
        assert _to_bytes_safe(b"ab") == b"ab"
        assert _to_bytes_safe(bytearray(b"xy")) == b"xy"
        assert _to_bytes_safe(memoryview(b"xy")) == b"xy"


class TestFdrBH:
    def test_all_significant(self):
        p = np.array([0.001, 0.002, 0.003])
        assert fdr_bh(p, alpha=0.05).all()

    def test_none_significant(self):
        p = np.array([0.5, 0.6, 0.9])
        assert not fdr_bh(p, alpha=0.05).any()

    def test_mixed_uses_step_up_cutoff(self):
        # thresh = 0.05*[1,2,3,4]/4 = [.0125,.025,.0375,.05]; sorted p = [.001,.01,.5,.8]
        # last passing rank is #2 (p=.01) -> cutoff .01 -> first two significant.
        p = np.array([0.001, 0.01, 0.5, 0.8])
        assert list(fdr_bh(p, alpha=0.05)) == [True, True, False, False]


class TestTemporalHoldoutSplit:
    def _xy(self):
        X = pd.DataFrame({"year": [2021, 2022, 2023, 2024, 2025], "f": [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 0])
        return X, y

    def test_splits_by_year_and_drops_year_col(self):
        X, y = self._xy()
        Xtr, Xva, ytr, yva = temporal_holdout_split(
            X, y, train_max_year=2022, valid_years=(2023, 2024)
        )
        assert "year" not in Xtr.columns and "year" not in Xva.columns
        assert list(ytr) == [0, 1]  # 2021, 2022
        assert list(yva) == [0, 1]  # 2023, 2024  (2025 excluded from both)
        assert len(Xtr) == 2 and len(Xva) == 2

    def test_overlap_raises(self):
        X, y = self._xy()
        with pytest.raises(AssertionError):
            temporal_holdout_split(X, y, train_max_year=2023, valid_years=(2023,))


class TestPrepareXyForLgb:
    def test_label_drops_and_type_handling(self):
        df = pd.DataFrame(
            {
                "y_next": [0, 1, 0],
                "hex_id": ["a", "b", "c"],  # drop_always
                "numstr": ["1", "2", "3"],  # numeric-convertible object
                "cat": ["x", "y", "x"],  # low-cardinality categorical
            }
        )
        X, y, cat_cols, diag = prepare_Xy_for_lgb(df)

        assert list(y) == [0, 1, 0]
        assert "y_next" not in X.columns and "hex_id" not in X.columns
        assert pd.api.types.is_numeric_dtype(X["numstr"])
        assert "cat" in cat_cols and str(X["cat"].dtype) == "category"
        assert diag["converted_to_num"] == ["numstr"]

    def test_missing_label_raises(self):
        with pytest.raises(KeyError):
            prepare_Xy_for_lgb(pd.DataFrame({"f": [1]}))


class TestBuildXLike:
    def test_aligns_to_reference_columns(self):
        raw = pd.DataFrame({"a": [1], "b": [2], "extra": [9], "y_next": [1]})
        out = build_X_like(
            raw,
            label_col="y_next",
            drop_always=["hex_id"],
            reference_columns=["a", "b", "c"],
            categorical_cols=[],
        )
        assert list(out.columns) == ["a", "b", "c"]  # order preserved, extra dropped
        assert out["c"].iloc[0] == 0.0  # missing ref filled with 0.0
        assert "y_next" not in out.columns


class TestFeatureMetaRoundTrip:
    def test_save_then_load(self, tmp_path):
        path = save_feature_meta(tmp_path, ["a", "b"], ["b"])
        meta = load_feature_meta(path)
        assert meta == {"feature_columns": ["a", "b"], "categorical_columns": ["b"]}


class TestSaveConfigSnapshot:
    def test_writes_resolved_yaml(self, tmp_path):
        cfg = OmegaConf.create({"model": {"n_estimators": 800}})
        path = save_config_snapshot(tmp_path, cfg)
        assert path.exists()
        assert "n_estimators" in path.read_text()
