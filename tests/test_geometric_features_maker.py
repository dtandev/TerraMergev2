import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from src.features.GeometricFeaturesMaker import (
    CompactnessCircularityFeatures,
    EdgeComplexityFeatures,
    EdgeContextFeatures,
    ElongationOrientationFeatures,
    ExtremeEnvelopeFeatures,
    MomentInertiaFeatures,
    SizeScaleFeatures,
)


def _gdf(poly, crs="EPSG:2180") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs=crs)


def _square(s=10.0) -> Polygon:
    return Polygon([(0, 0), (s, 0), (s, s), (0, s)])


def _rect(w, h) -> Polygon:
    return Polygon([(0, 0), (w, 0), (w, h), (0, h)])


def _l_shape() -> Polygon:
    # Concave "L": bounding box 10x10 with the top-right 6x6 removed.
    return Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])


class TestSizeScale:
    def test_square_features(self):
        out = SizeScaleFeatures().transform(_gdf(_square()))
        row = out.iloc[0]
        assert row["area_ha"] == pytest.approx(0.01)  # 100 m^2
        assert row["perimeter_m"] == pytest.approx(40.0)
        assert row["log_area"] == pytest.approx(math.log(100))
        assert row["mean_width"] == pytest.approx(5.0)  # 2*100/40

    def test_join_false_returns_only_features(self):
        out = SizeScaleFeatures(join=False).transform(_gdf(_square()))
        assert "id" not in out.columns
        assert "area_ha" in out.columns

    def test_missing_crs_raises(self):
        with pytest.raises(ValueError):
            SizeScaleFeatures().transform(_gdf(_square(), crs=None))

    def test_geographic_crs_raises(self):
        with pytest.raises(ValueError):
            SizeScaleFeatures().transform(_gdf(_square(), crs="EPSG:4326"))


class TestElongationOrientation:
    def test_rectangle_elongation_and_aspect(self):
        out = ElongationOrientationFeatures().transform(_gdf(_rect(20, 10)))
        row = out.iloc[0]
        assert row["elongation_mrr"] == pytest.approx(2.0)
        assert row["aspect_ratio_bbox"] == pytest.approx(2.0)

    def test_square_is_not_elongated(self):
        out = ElongationOrientationFeatures().transform(_gdf(_square()))
        assert out.iloc[0]["elongation_mrr"] == pytest.approx(1.0)


class TestCompactnessCircularity:
    def test_square_metrics(self):
        out = CompactnessCircularityFeatures().transform(_gdf(_square()))
        row = out.iloc[0]
        assert row["ipq"] == pytest.approx(math.pi / 4)  # 4*pi*100 / 40^2
        assert row["rectangularity"] == pytest.approx(1.0)
        assert row["solidity"] == pytest.approx(1.0)

    def test_concave_solidity_below_one(self):
        out = CompactnessCircularityFeatures().transform(_gdf(_l_shape()))
        assert out.iloc[0]["solidity"] < 1.0


class TestEdgeComplexity:
    def test_square_complexity_and_vertex_density(self):
        out = EdgeComplexityFeatures().transform(_gdf(_square()))
        row = out.iloc[0]
        assert row["complexity_index"] == pytest.approx(40 / (2 * math.sqrt(math.pi * 100)))
        assert row["vertex_density"] == pytest.approx(5 / 40)  # closed ring => 5 coords

    def test_count_vertices_handles_none(self):
        assert np.isnan(EdgeComplexityFeatures._count_vertices(None))


class TestMomentInertia:
    def test_ratio_is_finite_and_at_least_one(self):
        np.random.seed(0)  # _sample_boundary_points uses np.random
        out = MomentInertiaFeatures(sample_points=400).transform(_gdf(_rect(20, 5)))
        ratio = out.iloc[0]["inertia_ratio"]
        assert np.isfinite(ratio)
        assert ratio >= 1.0  # major/minor

    def test_empty_geometry_is_na(self):
        out = MomentInertiaFeatures().transform(_gdf(Polygon()))
        assert np.isnan(out.iloc[0]["inertia_ratio"])


class TestEdgeContext:
    def test_convex_square_has_zero_deficit(self):
        out = EdgeContextFeatures().transform(_gdf(_square()))
        row = out.iloc[0]
        assert row["convexity_deficit_area"] == pytest.approx(0.0, abs=1e-9)
        assert row["convexity_deficit_ratio"] == pytest.approx(0.0, abs=1e-9)

    def test_concave_shape_has_positive_deficit(self):
        out = EdgeContextFeatures().transform(_gdf(_l_shape()))
        assert out.iloc[0]["convexity_deficit_area"] > 0.0


class TestExtremeEnvelope:
    def test_square_radius_and_ratio(self):
        out = ExtremeEnvelopeFeatures().transform(_gdf(_square()))
        row = out.iloc[0]
        assert row["min_circle_radius"] == pytest.approx(math.hypot(5, 5))  # centroid->corner
        assert row["area_to_circle_ratio"] == pytest.approx(100 / (math.pi * 50))

    def test_max_radius_empty_is_na(self):
        assert np.isnan(ExtremeEnvelopeFeatures._max_radius(Polygon()))
