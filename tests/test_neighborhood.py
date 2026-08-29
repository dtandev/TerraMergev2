import h3
import pandas as pd
import pytest

from src.modeling.neighborhood import (
    _most_frequent_non_null,
    _neighbors_disk,
    _neighbors_exact_ring,
    _to_bytes_safe,
    build_h3_neighbors_edges,
    compute_neighbor_aggregates,
)

# A hexagon (non-pentagon) cell over Warsaw at res 7 → exactly 6 immediate neighbors.
CENTER = h3.geo_to_h3(52.0, 21.0, 7)
NEIGHBORS = sorted(h3.k_ring(CENTER, 1) - {CENTER})


class TestToBytesSafe:
    def test_variants(self):
        assert _to_bytes_safe(None) is None
        assert _to_bytes_safe(bytearray(b"xy")) == b"xy"
        assert _to_bytes_safe(memoryview(b"xy")) == b"xy"


class TestMostFrequentNonNull:
    def test_returns_mode(self):
        assert _most_frequent_non_null(pd.Series(["x", "x", "y", None])) == "x"

    def test_all_null_returns_none(self):
        assert _most_frequent_non_null(pd.Series([None, None], dtype=object)) is None


class TestH3Neighbors:
    def test_disk_ring1_is_six(self):
        assert len(_neighbors_disk(CENTER, 1)) == 6
        assert CENTER not in _neighbors_disk(CENTER, 1)

    def test_exact_ring2_is_twelve(self):
        assert len(_neighbors_exact_ring(CENTER, 2)) == 12

    def test_disk_ring2_is_eighteen(self):
        assert len(_neighbors_disk(CENTER, 2)) == 18  # ring1 (6) + ring2 (12)

    def test_ring_below_one_is_empty(self):
        assert _neighbors_exact_ring(CENTER, 0) == []


class TestBuildEdges:
    def test_edges_both_directions_within_cell_set(self):
        n0 = NEIGHBORS[0]
        edges = build_h3_neighbors_edges([CENTER, n0], R_values=1, rolling=True)
        pairs = set(zip(edges["cell"], edges["neighbor"]))
        assert (CENTER, n0) in pairs
        assert (n0, CENTER) in pairs
        assert list(edges.columns) == ["cell", "neighbor", "R"]

    def test_no_rings_returns_empty(self):
        out = build_h3_neighbors_edges([CENTER], R_values=[0], rolling=True)
        assert out.empty


class TestComputeNeighborAggregates:
    def _df(self):
        rows = [{"hex_id": CENTER, "year": 2020, "feat": 100.0, "cat": "C"}]
        feats = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        cats = ["A", "A", "A", "A", "B", "B"]
        for n, fv, cv in zip(NEIGHBORS, feats, cats):
            rows.append({"hex_id": n, "year": 2020, "feat": fv, "cat": cv})
        return pd.DataFrame(rows)

    def test_center_aggregates_over_its_six_neighbors(self):
        out = compute_neighbor_aggregates(self._df(), R_values=1, categorical_cols=("cat",))
        crow = out[out["hex_id"] == CENTER].iloc[0]
        assert crow["nbr_r1_n"] == 6
        assert crow["nbr_r1_feat_mean"] == pytest.approx(3.5)  # mean(1..6), self excluded
        assert crow["nbr_r1_feat_median"] == pytest.approx(3.5)
        assert crow["nbr_r1_cat_mode"] == "A"  # 4x A vs 2x B

    def test_duplicate_keys_raise(self):
        df = pd.DataFrame({"hex_id": [CENTER, CENTER], "year": [2020, 2020], "feat": [1.0, 2.0]})
        with pytest.raises(ValueError):
            compute_neighbor_aggregates(df, R_values=1)
