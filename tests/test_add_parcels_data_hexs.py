import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from src.features.add_parcels_data_hexs import (
    _postprocess_coverage_and_rounding,
    _to_bytes_safe,
    intersect_aggregate_hex_parcels,
)


def _rect(x0, y0, x1, y1) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _hex() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"hex_id": ["h1"]}, geometry=[_rect(0, 0, 10, 10)], crs="EPSG:2180")


def _parcels(geoms=None, **cols) -> gpd.GeoDataFrame:
    data = {
        "iddzialki": cols.get("iddzialki", ["a", "b"]),
        "year": cols.get("year", [2020, 2020]),
        "feat": cols.get("feat", [10.0, 20.0]),
        "jednostka": cols.get("jednostka", ["X", "Y"]),
    }
    return gpd.GeoDataFrame(
        data,
        geometry=geoms or [_rect(0, 0, 10, 6), _rect(0, 6, 10, 10)],
        crs="EPSG:2180",
    )


class TestToBytesSafe:
    def test_none(self):
        assert _to_bytes_safe(None) is None

    def test_bytes_passthrough(self):
        assert _to_bytes_safe(b"abc") == b"abc"

    def test_bytearray_and_memoryview(self):
        assert _to_bytes_safe(bytearray(b"xy")) == b"xy"
        assert _to_bytes_safe(memoryview(b"xy")) == b"xy"

    def test_list_of_ints(self):
        assert _to_bytes_safe([104, 105]) == b"hi"

    def test_unhandled_types_return_none(self):
        assert _to_bytes_safe("hex-string") is None
        assert _to_bytes_safe([1, 2, 999]) is None  # 999 out of byte range


class TestIntersectAggregateHexParcels:
    def test_area_weighted_mean_dominant_and_counts(self):
        out = intersect_aggregate_hex_parcels(_parcels(), _hex()).set_index("hex_id")
        row = out.loc["h1"]
        assert row["feat_mean"] == pytest.approx(14.0)  # (10*0.6 + 20*0.4)/1.0
        assert row["n_parcel"] == 2
        assert row["jednostka"] == "X"  # 60 m^2 > 40 m^2
        assert row["hex_area"] == pytest.approx(100.0)
        assert row["coverage_area"] == pytest.approx(100.0)

    def test_n_parcel_counts_unique_ids(self):
        # Two parcel parts share the same id "a" → counts once, plus "b".
        p = _parcels(
            geoms=[_rect(0, 0, 5, 10), _rect(5, 0, 8, 10), _rect(8, 0, 10, 10)],
            iddzialki=["a", "a", "b"],
            year=[2020, 2020, 2020],
            feat=[10.0, 10.0, 20.0],
            jednostka=["X", "X", "Y"],
        )
        out = intersect_aggregate_hex_parcels(p, _hex()).set_index("hex_id")
        assert out.loc["h1", "n_parcel"] == 2

    def test_empty_when_no_intersection(self):
        p = _parcels(geoms=[_rect(100, 100, 101, 101), _rect(102, 102, 103, 103)])
        assert intersect_aggregate_hex_parcels(p, _hex()).empty

    def test_missing_parcel_id_raises(self):
        p = _parcels().drop(columns=["iddzialki"])
        with pytest.raises(ValueError):
            intersect_aggregate_hex_parcels(p, _hex())

    def test_hex_without_crs_raises(self):
        h = gpd.GeoDataFrame({"hex_id": ["h1"]}, geometry=[_rect(0, 0, 10, 10)], crs=None)
        with pytest.raises(ValueError):
            intersect_aggregate_hex_parcels(_parcels(), h)


class TestPostprocessCoverageAndRounding:
    def _out(self, coverage=50.0, hex_area=100.0):
        return gpd.GeoDataFrame(
            {
                "hex_id": ["h1"],
                "feat_mean": [14.123456],
                "hex_area": [hex_area],
                "coverage_area": [coverage],
            },
            geometry=[_rect(0, 0, 10, 10)],
            crs="EPSG:2180",
        )

    def test_adds_coverage_frac_and_rounds(self):
        out = _postprocess_coverage_and_rounding(self._out(), min_cover_fraction=0.0, decimals=2)
        assert out.iloc[0]["coverage_frac"] == pytest.approx(0.5)
        assert out.iloc[0]["feat_mean"] == pytest.approx(14.12)

    def test_min_cover_fraction_filters_rows(self):
        out = _postprocess_coverage_and_rounding(
            self._out(coverage=5.0), min_cover_fraction=0.5, decimals=2
        )
        assert out.empty

    def test_zero_hex_area_rows_dropped(self):
        out = _postprocess_coverage_and_rounding(
            self._out(hex_area=0.0), min_cover_fraction=0.0, decimals=2
        )
        assert out.empty
