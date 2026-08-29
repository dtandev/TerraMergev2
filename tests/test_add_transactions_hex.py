import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from src.features.add_transactions_hex import (
    _apply_prefix,
    _dedupe_by_keys,
    intersect_and_aggregate_area_weighted,
)


def _rect(x0, y0, x1, y1) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _hex() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"hex_id": ["h1"], "hex_area_m2": [100.0]},
        geometry=[_rect(0, 0, 10, 10)],
        crs="EPSG:2180",
    )


class TestDedupeByKeys:
    def test_first_keeps_first_per_key(self):
        df = pd.DataFrame({"k": [1, 1, 2], "v": ["a", "b", "c"]})
        out = _dedupe_by_keys(df, ["k"], strategy="first")
        assert sorted(out["v"]) == ["a", "c"]

    def test_last_with_order_by(self):
        df = pd.DataFrame({"k": [1, 1], "ord": [5, 9], "v": ["a", "b"]})
        out = _dedupe_by_keys(df, ["k"], strategy="last", order_by="ord")
        # last => sorted descending, first row kept => the higher 'ord'
        assert out.iloc[0]["v"] == "b"

    def test_reduce_applies_reduce_map_and_defaults(self):
        df = pd.DataFrame({"k": [1, 1], "cena": [10, 30], "area": [2.0, 4.0], "name": ["x", "y"]})
        out = _dedupe_by_keys(df, ["k"], strategy="reduce", reduce_map={"cena": "max"})
        row = out.iloc[0]
        assert row["cena"] == 30  # reduce_map max
        assert row["area"] == pytest.approx(3.0)  # default mean
        assert row["name"] == "x"  # non-numeric first


class TestApplyPrefix:
    def test_prefixes_non_key_columns(self):
        df = pd.DataFrame({"k": [1], "a": [2], "b": [3]})
        out = _apply_prefix(df, "tx_", ["k"])
        assert set(out.columns) == {"k", "tx_a", "tx_b"}

    def test_empty_prefix_is_noop(self):
        df = pd.DataFrame({"k": [1], "a": [2]})
        assert list(_apply_prefix(df, "", ["k"]).columns) == ["k", "a"]


class TestIntersectAndAggregate:
    def _left(self, **over):
        # Bottom 60% is unit X value 10, top 40% is unit Y value 20.
        data = {
            "year": [2020, 2020],
            "jednostka": ["X", "Y"],
            "value": over.get("value", [10.0, 20.0]),
        }
        return gpd.GeoDataFrame(
            data,
            geometry=[_rect(0, 0, 10, 6), _rect(0, 6, 10, 10)],
            crs="EPSG:2180",
        )

    def test_area_weighted_mean_and_dominant_category(self):
        out = intersect_and_aggregate_area_weighted(self._left(), _hex()).set_index("hex_id")
        assert out.loc["h1", "value_mean"] == pytest.approx(14.0)  # (10*60 + 20*40)/100
        assert out.loc["h1", "jednostka"] == "X"  # 60 m^2 beats 40
        assert out.loc["h1", "year"] == 2020

    def test_empty_when_no_intersection(self):
        left = self._left()
        left["geometry"] = [_rect(100, 100, 101, 101), _rect(102, 102, 103, 103)]
        out = intersect_and_aggregate_area_weighted(left, _hex())
        assert out.empty

    def test_min_cover_fraction_filters_small_overlaps(self):
        # Each parcel covers <5% of the hex; a 0.5 threshold drops everything.
        left = self._left()
        left["geometry"] = [_rect(0, 0, 1, 1), _rect(1, 1, 2, 2)]
        out = intersect_and_aggregate_area_weighted(left, _hex(), min_cover_fraction=0.5)
        assert out.empty

    def test_treat_zero_as_na_and_nonzero_mean(self):
        left = self._left(value=[0.0, 20.0])  # X contributes a zero
        out = intersect_and_aggregate_area_weighted(
            left,
            _hex(),
            treat_zero_as_na=["value"],
            extra_nonzero_mean_cols=["value"],
        ).set_index("hex_id")
        # plain mean divides the nonzero numerator by the FULL area: (20*40)/100 = 8.0
        assert out.loc["h1", "value_mean"] == pytest.approx(8.0)
        # nonzero mean divides only by the nonzero area: (20*40)/40 = 20.0
        assert out.loc["h1", "value_mean_nz"] == pytest.approx(20.0)
