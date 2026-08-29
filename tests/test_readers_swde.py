from src.prepare_data.readers.swde_reader import (
    _build_kkl_polygons,
    _build_parcel_polygons,
    _build_point_dict,
    _parse_records,
    _swde_type,
    read_all_layers,
)

_EWMAPA_FIXTURE = """SWDE.w.2.00.(C) GUGiK 2000;
SN;
NS,ZN,EWMAPA
SX;
SO;
RP,,G5PZG,P1,1,11;
P,G,0.0,0.0,;
X;
RP,,G5PZG,P2,2,11;
P,G,1.0,0.0,;
X;
RP,,G5PZG,P3,3,11;
P,G,1.0,1.0,;
X;
RO,,G5G_DZE,1-1,1,11;
GL;
K,+;
P,P,G5PZG,P1;
P,P,G5PZG,P2;
P,P,G5PZG,P3;
PZ;
GX;
D,G5IDD,D,999901_1.0001.1
X;
"""

_EWOPIS_FIXTURE = """SWDE.w.2.00.(C) GUGiK 2000;
SN;
NS,ZN,EWOPIS
SX;
SO;
RO,,G5DZE,1,1,11;
D,G5IDD,D,999901_1.0001.1
D,G5PEW,D,1000
X;
"""

# Real case (rok_2021/2023 szczycieński): a file that declares NS,ZN,EWOPIS in its header but
# still carries full ring geometry for a parcel -- same record shape as _EWMAPA_FIXTURE, just
# with the mismatched declared type.
_MISLABELED_EWOPIS_WITH_GEOMETRY_FIXTURE = _EWMAPA_FIXTURE.replace("NS,ZN,EWMAPA", "NS,ZN,EWOPIS")

# Real case, same delivery as above: the parcel geometry+attributes record uses the plain "G5DZE"
# table (no middle "G_") instead of the standard "G5G_DZE" -- confirmed on real data this is a
# second, independent quirk of the same non-standard vendor export (on top of the mislabeled
# NS,ZN,EWOPIS type), not an alternate scenario.
_MISLABELED_EWOPIS_PLAIN_G5DZE_TABLE_FIXTURE = _MISLABELED_EWOPIS_WITH_GEOMETRY_FIXTURE.replace(
    "G5G_DZE", "G5DZE"
)

# Same parcel as _EWMAPA_FIXTURE, plus one G5KKL contour record: self-contained inline-coordinate
# ring (no G5PZG lookup) carrying both land-use (G5OZU) and classification (G5OZK) attributes —
# confirmed on real data to always carry both on the same record.
_EWMAPA_WITH_KKL_FIXTURE = (
    _EWMAPA_FIXTURE
    + """RO,,G5KKL,12-1,101,11;
GL;
K,+;
P,G,0.0,0.0,;
P,G,1.0,0.0,;
P,G,1.0,1.0,;
PZ;
GX;
D,G5IDK,D,999901_1.0012.KL.1
D,G5OZU,D,R
D,G5OZK,D,IVa
X;
"""
)

# Same as above but the contour table uses the alternate real-world name "G5G_KKL" instead of
# "G5KKL" — both names are seen across real deliveries.
_EWMAPA_WITH_G5G_KKL_FIXTURE = _EWMAPA_WITH_KKL_FIXTURE.replace("G5KKL", "G5G_KKL")

# Real case (rok_2013 elblaski): a G5G_DZE parcel ring given as inline P,G, coordinates directly
# (no G5PZG point-dictionary lookup), the same style G5KKL always uses -- rather than the standard
# P,P, point-id references _EWMAPA_FIXTURE uses. Confirmed on real data: the parcel's own rings
# already contain (x, y) tuples, so treating them as point-id strings (the old, only-supported
# path) silently dropped every parcel with this ring style.
_EWMAPA_WITH_INLINE_COORD_PARCEL_RING_FIXTURE = """SWDE.w.2.00.(C) GUGiK 2000;
SN;
NS,ZN,EWMAPA
SX;
SO;
RO,,G5G_DZE,1-1,1,11;
GL;
K,+;
P,G,0.0,0.0,;
P,G,1.0,0.0,;
P,G,1.0,1.0,;
PZ;
GX;
D,G5IDD,D,999901_1.0001.1
X;
"""

# Same content as _EWMAPA_WITH_KKL_FIXTURE, but every RECORD ends with "XC,<checksum>;" instead of
# "X;" — a real alternative terminator confirmed on a real rok_2016 ostródzki delivery, used for
# every table in that file, not just one. Line-exact replacement (not a blind string .replace)
# since "X;" is also a substring of the unrelated header marker "SX;".
_EWMAPA_XC_TERMINATOR_FIXTURE = "\n".join(
    "XC,123456;" if line == "X;" else line for line in _EWMAPA_WITH_KKL_FIXTURE.split("\n")
)

# Real case (rok_2020 elblaski): the point dictionary's own record header carries a secondary
# "alias" id ("PZG_1", the 5th comma field) alongside its primary GUID id, and the G5DZE parcel
# ring references that alias via a `P,K,` marker instead of the standard `P,P,` (which everywhere
# else references the PRIMARY id). Without recognizing `P,K,` at all, and without indexing the
# point dict by the alias id too, every ring in this delivery style resolved to zero points.
_EWMAPA_WITH_ALIAS_ID_POINT_REF_FIXTURE = """SWDE.w.2.00.(C) GUGiK 2000;
SN;
NS,ZN,EWMAPA
SX;
SO;
RP,,G5PZG,GUID-P1,PZG_1,11;
P,G,0.0,0.0,;
X;
RP,,G5PZG,GUID-P2,PZG_2,11;
P,G,1.0,0.0,;
X;
RP,,G5PZG,GUID-P3,PZG_3,11;
P,G,1.0,1.0,;
X;
RO,,G5G_DZE,1-1,1,11;
GL;
K,+;
P,K,PZG_1;
P,K,PZG_2;
P,K,PZG_3;
PZ;
GX;
D,G5IDD,D,999901_1.0001.1
X;
"""


class TestSwdeType:
    def test_detects_ewmapa(self):
        assert _swde_type(_EWMAPA_FIXTURE.split("\n")) == "EWMAPA"

    def test_detects_ewopis(self):
        assert _swde_type(_EWOPIS_FIXTURE.split("\n")) == "EWOPIS"


class TestParseRecords:
    def test_parses_point_and_geometry_records(self):
        records = _parse_records(_EWMAPA_FIXTURE.split("\n"))
        assert len(records["G5PZG"]) == 3
        assert records["G5PZG"][0]["_id"] == "P1"
        assert records["G5PZG"][0]["_xy"] == (0.0, 0.0)

        assert len(records["G5G_DZE"]) == 1
        rec = records["G5G_DZE"][0]
        assert rec["G5IDD"] == "999901_1.0001.1"
        assert rec["_rings"] == [("+", ["P1", "P2", "P3"])]


class TestParseRecordsKkl:
    def test_inline_ring_coords_appended_as_tuples_not_overwritten(self):
        # Regression guard: before the fix, each P,G, line overwrote a single "_xy" slot instead
        # of appending to the ring, so a multi-vertex inline ring silently lost every vertex but
        # the last one.
        records = _parse_records(_EWMAPA_WITH_KKL_FIXTURE.split("\n"))
        rec = records["G5KKL"][0]
        assert rec["_rings"] == [("+", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])]
        assert rec["G5OZU"] == "R"
        assert rec["G5OZK"] == "IVa"

    def test_xc_terminator_commits_records_same_as_x(self):
        records = _parse_records(_EWMAPA_XC_TERMINATOR_FIXTURE.split("\n"))
        assert len(records["G5G_DZE"]) == 1
        assert len(records["G5KKL"]) == 1
        assert records["G5KKL"][0]["G5OZU"] == "R"


class TestBuildKklPolygons:
    def test_reconstructs_contour_with_land_use_and_classification(self):
        records = _parse_records(_EWMAPA_WITH_KKL_FIXTURE.split("\n"))
        gdf = _build_kkl_polygons(records)

        assert len(gdf) == 1
        assert gdf.iloc[0]["G5OZU"] == "R"
        assert gdf.iloc[0]["G5OZK"] == "IVa"
        assert gdf.iloc[0]["G5IDK"] == "999901_1.0012.KL.1"
        geom = gdf.geometry.iloc[0]
        assert geom.is_valid
        assert geom.area == 0.5

    def test_recognizes_alternate_table_name_g5g_kkl(self):
        records = _parse_records(_EWMAPA_WITH_G5G_KKL_FIXTURE.split("\n"))
        gdf = _build_kkl_polygons(records)
        assert len(gdf) == 1

    def test_empty_when_no_kkl_table_present(self):
        records = _parse_records(_EWMAPA_FIXTURE.split("\n"))
        gdf = _build_kkl_polygons(records)
        assert gdf.empty


class TestBuildParcelPolygons:
    def test_reconstructs_triangle_from_point_refs(self):
        records = _parse_records(_EWMAPA_FIXTURE.split("\n"))
        points = _build_point_dict(records)
        gdf = _build_parcel_polygons(records, points)

        assert len(gdf) == 1
        assert gdf.iloc[0]["idDzialki"] == "999901_1.0001.1"
        assert gdf.crs.to_epsg() == 2178  # default zone-7 CRS
        geom = gdf.geometry.iloc[0]
        assert geom.is_valid
        assert geom.area == 0.5  # right triangle (0,0)-(1,0)-(1,1)

    def test_crs_is_configurable_not_hardcoded(self):
        # SWDE carries no CRS metadata of its own — the default (EPSG:2178) is only correct for
        # szczycieński/zone 7. A future run against a different powiat/zone must be able to
        # override it via the `crs` param rather than editing this module.
        records = _parse_records(_EWMAPA_FIXTURE.split("\n"))
        points = _build_point_dict(records)
        gdf = _build_parcel_polygons(records, points, crs="EPSG:2177")

        assert gdf.crs.to_epsg() == 2177


class TestReadAllLayers:
    def test_ewmapa_file_yields_dzialka_ewidencyjna_layer(self, tmp_path):
        path = tmp_path / "999901_1.swd"
        path.write_text(_EWMAPA_FIXTURE, encoding="iso-8859-2")

        layers = read_all_layers(path)

        assert set(layers.keys()) == {"DzialkaEwidencyjna"}
        assert len(layers["DzialkaEwidencyjna"]) == 1
        assert layers["DzialkaEwidencyjna"].crs.to_epsg() == 2178

    def test_read_all_layers_honors_crs_override(self, tmp_path):
        path = tmp_path / "999901_1.swd"
        path.write_text(_EWMAPA_FIXTURE, encoding="iso-8859-2")

        layers = read_all_layers(path, crs="EPSG:2176")

        assert layers["DzialkaEwidencyjna"].crs.to_epsg() == 2176

    def test_ewopis_file_yields_no_layers(self, tmp_path):
        path = tmp_path / "999901_1.swd"
        path.write_text(_EWOPIS_FIXTURE, encoding="iso-8859-2")

        layers = read_all_layers(path)

        assert layers == {}

    def test_mislabeled_ewopis_with_real_geometry_is_still_read(self, tmp_path):
        path = tmp_path / "999901_1.swd"
        path.write_text(_MISLABELED_EWOPIS_WITH_GEOMETRY_FIXTURE, encoding="iso-8859-2")

        layers = read_all_layers(path)

        assert set(layers.keys()) == {"DzialkaEwidencyjna"}
        assert len(layers["DzialkaEwidencyjna"]) == 1

    def test_plain_g5dze_table_name_is_used_for_geometry_when_g5g_dze_absent(self, tmp_path):
        path = tmp_path / "999901_1.swd"
        path.write_text(_MISLABELED_EWOPIS_PLAIN_G5DZE_TABLE_FIXTURE, encoding="iso-8859-2")

        layers = read_all_layers(path)

        assert set(layers.keys()) == {"DzialkaEwidencyjna"}
        assert len(layers["DzialkaEwidencyjna"]) == 1

    def test_parcel_ring_with_inline_coordinates_is_reconstructed(self, tmp_path):
        path = tmp_path / "999901_1.swd"
        path.write_text(_EWMAPA_WITH_INLINE_COORD_PARCEL_RING_FIXTURE, encoding="iso-8859-2")

        layers = read_all_layers(path)

        assert set(layers.keys()) == {"DzialkaEwidencyjna"}
        assert len(layers["DzialkaEwidencyjna"]) == 1

    def test_ring_reference_to_point_alias_id_is_resolved(self, tmp_path):
        path = tmp_path / "999901_1.swd"
        path.write_text(_EWMAPA_WITH_ALIAS_ID_POINT_REF_FIXTURE, encoding="iso-8859-2")

        layers = read_all_layers(path)

        assert set(layers.keys()) == {"DzialkaEwidencyjna"}
        assert len(layers["DzialkaEwidencyjna"]) == 1

    def test_missing_file_returns_empty(self, tmp_path):
        layers = read_all_layers(tmp_path / "does_not_exist.swd")
        assert layers == {}

    def test_kkl_table_yields_uzg_and_kkl_layers_alongside_parcels(self, tmp_path):
        path = tmp_path / "999901_1.swd"
        path.write_text(_EWMAPA_WITH_KKL_FIXTURE, encoding="iso-8859-2")

        layers = read_all_layers(path)

        assert set(layers.keys()) == {
            "DzialkaEwidencyjna",
            "KonturUzytkuGruntowego",
            "KonturKlasyfikacyjny",
        }
        assert len(layers["KonturUzytkuGruntowego"]) == 1
        assert layers["KonturUzytkuGruntowego"].iloc[0]["G5OZU"] == "R"
        assert layers["KonturKlasyfikacyjny"].iloc[0]["G5OZK"] == "IVa"

    def test_xc_terminator_file_reads_same_as_x(self, tmp_path):
        path = tmp_path / "999901_1.swd"
        path.write_text(_EWMAPA_XC_TERMINATOR_FIXTURE, encoding="iso-8859-2")

        layers = read_all_layers(path)

        assert set(layers.keys()) == {
            "DzialkaEwidencyjna",
            "KonturUzytkuGruntowego",
            "KonturKlasyfikacyjny",
        }
