import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon

from src.prepare_data.readers.shp_reader import (
    _alias_to_canonical,
    _resolve_unresolvable_crs,
    read_all_layers,
)

# Real ESRI-flavored `.prj` WKT for Polish CS2000 zone 7 (central meridian 21), taken verbatim
# from a real EGiB SHP delivery's G5G_DZE.prj. pyproj's CRS.to_epsg() returns None for this exact
# string (confirmed) — it's the whole reason _resolve_unresolvable_crs exists.
_LEGACY_ESRI_PRJ_ZONE_7 = (
    'PROJCS["ETRS_1989_UWPP_2000_PAS_7",GEOGCS["GCS_ETRS_1989",'
    'DATUM["D_ETRS_1989",SPHEROID["GRS_1980",6378137.0,298.257222101]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",7500000.0],'
    'PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",21.0],'
    'PARAMETER["Scale_Factor",0.999923],PARAMETER["Latitude_Of_Origin",0.0],'
    'UNIT["Meter",1.0]]'
)

LAYER_NAME_MAP = {
    "DzialkaEwidencyjna": ["EGB_DzialkaEwidencyjna", "G5G_DZE"],
    "Budynek": ["EGB_Budynek", "G5G_BUD"],
}


def _fixture_gdf(geom=None) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"G5IDD": ["281701_1.0001.1"], "SHAPE_Leng": [4.0]},
        geometry=[geom or Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:2180",
    )


class TestAliasToCanonical:
    def test_inverts_map_case_insensitively(self):
        alias_map = _alias_to_canonical(LAYER_NAME_MAP)
        assert alias_map["G5G_DZE"] == ["DzialkaEwidencyjna"]
        assert alias_map["EGB_BUDYNEK"] == ["Budynek"]

    def test_alias_can_feed_multiple_canonical_layers(self):
        # Real case: G5G_KKL carries both a land-use (G5OZU) and classification (G5OZK)
        # attribute on the same geometry, so it legitimately maps to both layers.
        layer_name_map = {
            "KonturKlasyfikacyjny": ["G5G_KKL"],
            "KonturUzytkuGruntowego": ["G5G_UZG", "G5G_KKL"],
        }
        alias_map = _alias_to_canonical(layer_name_map)
        assert alias_map["G5G_KKL"] == ["KonturKlasyfikacyjny", "KonturUzytkuGruntowego"]


class TestResolveUnresolvableCrs:
    def test_maps_pas_7_wkt_to_epsg_2178(self):
        gdf = _fixture_gdf()
        gdf = gdf.set_crs(CRS.from_wkt(_LEGACY_ESRI_PRJ_ZONE_7), allow_override=True)
        assert gdf.crs.to_epsg() is None  # sanity check: this WKT is indeed unresolvable

        out = _resolve_unresolvable_crs(gdf)

        assert out.crs.to_epsg() == 2178

    def test_noop_when_crs_already_resolves(self):
        gdf = _fixture_gdf()  # already EPSG:2180
        out = _resolve_unresolvable_crs(gdf)
        assert out.crs.to_epsg() == 2180

    def test_noop_when_crs_is_none(self):
        gdf = _fixture_gdf().set_crs(None, allow_override=True)
        out = _resolve_unresolvable_crs(gdf)
        assert out.crs is None


class TestReadAllLayers:
    def test_recognized_shp_mapped_to_canonical_layer(self, tmp_path):
        shp_path = tmp_path / "G5G_DZE.shp"
        _fixture_gdf().to_file(shp_path)

        layers = read_all_layers([shp_path], LAYER_NAME_MAP)

        assert set(layers.keys()) == {"DzialkaEwidencyjna"}
        gdf = layers["DzialkaEwidencyjna"]
        assert gdf.iloc[0]["G5IDD"] == "281701_1.0001.1"

    def test_unrecognized_shp_is_skipped(self, tmp_path):
        shp_path = tmp_path / "G5_UNKNOWN_LAYER.shp"
        _fixture_gdf().to_file(shp_path)

        layers = read_all_layers([shp_path], LAYER_NAME_MAP)

        assert layers == {}

    def test_non_polygon_shp_is_skipped(self, tmp_path):
        from shapely.geometry import Point

        shp_path = tmp_path / "G5G_DZE.shp"
        _fixture_gdf(geom=Point(0, 0)).to_file(shp_path)

        layers = read_all_layers([shp_path], LAYER_NAME_MAP)

        assert layers == {}

    def test_multiple_units_layers_combined(self, tmp_path):
        dze_path = tmp_path / "G5G_DZE.shp"
        bud_path = tmp_path / "G5G_BUD.shp"
        _fixture_gdf().to_file(dze_path)
        _fixture_gdf().to_file(bud_path)

        layers = read_all_layers([dze_path, bud_path], LAYER_NAME_MAP)

        assert set(layers.keys()) == {"DzialkaEwidencyjna", "Budynek"}

    def test_kkl_shp_feeds_both_uzg_and_kkl_layers(self, tmp_path):
        # Real case: G5G_KKL.shp is the only geometry-bearing contour file in many legacy SHP
        # deliveries — G5G_UZG.shp itself is often missing (attribute-only .dbf, no shape).
        layer_name_map = {
            "KonturKlasyfikacyjny": ["G5G_KKL"],
            "KonturUzytkuGruntowego": ["G5G_UZG", "G5G_KKL"],
        }
        kkl_path = tmp_path / "G5G_KKL.shp"
        _fixture_gdf().to_file(kkl_path)

        layers = read_all_layers([kkl_path], layer_name_map)

        assert set(layers.keys()) == {"KonturKlasyfikacyjny", "KonturUzytkuGruntowego"}
        assert layers["KonturKlasyfikacyjny"].iloc[0]["G5IDD"] == "281701_1.0001.1"
        assert layers["KonturUzytkuGruntowego"].iloc[0]["G5IDD"] == "281701_1.0001.1"
