import pandas as pd

from src.prepare_data.extract_polygons import (
    UNIT_PATTERN,
    _combined_powiat_prefix,
    _discover_units,
    _jew_layer_name,
    _read_unit_layers,
    _unit_code_from_match,
)


class _FakeLayer:
    def __init__(self, name):
        self._name = name

    def GetName(self):
        return self._name


class _FakeDataSource:
    def __init__(self, layer_names):
        self._layers = [_FakeLayer(n) for n in layer_names]

    def GetLayerCount(self):
        return len(self._layers)

    def GetLayerByIndex(self, i):
        return self._layers[i]


class TestUnitPattern:
    def test_matches_any_district_not_just_2815(self):
        # Regression guard: UNIT_PATTERN used to be hardcoded to powiat 2815 (ostródzki) only,
        # silently breaking unit-code detection for every other district (e.g. 2817 szczycieński).
        assert _unit_code_from_match(UNIT_PATTERN.search("281701_1_M_Szczytno.gdb")) == "281701_1"
        assert _unit_code_from_match(UNIT_PATTERN.search("281501_2_Gm_Olecko.gdb")) == "281501_2"
        assert _unit_code_from_match(UNIT_PATTERN.search("100201_3.gml")) == "100201_3"

    def test_does_not_false_positive_inside_longer_digit_runs(self):
        # Regression guard: found on real data (rok_2017, powiat 2802 braniewski) — a delivery
        # folder named with a date stamp + powiat code, "20170303_2802_urzad_marszalkowski",
        # used to match as a fake unit "170303_2" because the old pattern had no digit-boundary
        # guard. A real TERYT unit code is never embedded inside a longer run of digits.
        assert UNIT_PATTERN.search("20170303_2802_urzad_marszalkowski") is None

    def test_matches_compact_7_digit_form_without_underscore(self):
        # Regression guard: found on real data across 8 powiats (2802, 2808, 2810, 2814, 2815,
        # 2818, 2819, 2862) — some deliveries write the same TERYT code as 7 digits with no
        # separator (e.g. "2814014.GML" = powiat 2814 + gmina 01 + rodzaj 4) instead of the usual
        # "281401_4". Without this branch, entire deliveries (e.g. powiat olsztyński 2023-2026)
        # were silently invisible to discovery — reported as "no data" when data existed.
        assert _unit_code_from_match(UNIT_PATTERN.search("2814014.GML")) == "281401_4"
        assert _unit_code_from_match(UNIT_PATTERN.search("2862011.gml")) == "286201_1"

    def test_compact_form_rejects_non_28_powiat_prefix(self):
        # Regression guard: found on real data (rok_2021, powiat 2804 elbląski) — auxiliary
        # BDOT/GESUT layers bundled into the same GML delivery as EGiB have their own internal
        # numeric IDs ("BDOT_4000003.gml", "GESUT_4000004.gml") that coincidentally match the
        # compact 7-digit pattern. Since every real file in this dataset is confirmed to belong to
        # woj. warmińsko-mazurskie (TERYT prefix 28), restricting the compact form to that prefix
        # rejects these without narrowing the underscore-delimited branch (which stays general).
        assert UNIT_PATTERN.search("BDOT_4000003.gml") is None
        assert UNIT_PATTERN.search("GESUT_4000004.gml") is None

    def test_compact_form_rejects_invalid_rodzaj_digit(self):
        # TERYT's "rodzaj gminy" digit is only ever 1-5 — a trailing 6-9 means this 7-digit run
        # is not a unit code at all, just some other number, so it must not match.
        assert UNIT_PATTERN.search("2814019.GML") is None


class TestDiscoverUnits:
    def test_finds_units_across_all_formats(self, tmp_path):
        year_dir = tmp_path / "rok_2099"
        (year_dir / "gdb" / "281701_1_szczytno.gdb").mkdir(parents=True)
        (year_dir / "gml").mkdir(parents=True)
        (year_dir / "gml" / "281702_2.gml").touch()
        shp_dir = year_dir / "SHP" / "281703_2_jedwabno"
        shp_dir.mkdir(parents=True)
        (shp_dir / "G5G_DZE.shp").touch()
        (year_dir / "swde").mkdir(parents=True)
        (year_dir / "swde" / "281704_4.swd").touch()

        units = _discover_units(year_dir)

        assert "281701_1" in units and "gdb" in units["281701_1"]
        assert "281702_2" in units and "gml" in units["281702_2"]
        assert "281703_2" in units and "shp" in units["281703_2"]
        assert "281704_4" in units and "swde" in units["281704_4"]

    def test_no_units_found_returns_empty(self, tmp_path):
        year_dir = tmp_path / "rok_2099"
        year_dir.mkdir()
        assert _discover_units(year_dir) == {}

    def test_finds_4_letter_swde_extension(self, tmp_path):
        # Regression guard: found on real data (260 files across the dataset, e.g.
        # rok_2020/2815_ostrodzki/SWDE/opisowa/281501_1.swde) — some deliveries use the
        # 4-letter ".swde" extension for the exact same file format as ".swd". Discovery only
        # globbed "*.swd"/"*.SWD", silently dropping whole deliveries that used ".swde".
        year_dir = tmp_path / "rok_2099"
        (year_dir / "swde").mkdir(parents=True)
        (year_dir / "swde" / "281501_1.swde").touch()

        units = _discover_units(year_dir)

        assert "281501_1" in units and "swde" in units["281501_1"]


class TestReadUnitLayers:
    def test_prefers_gdb_over_everything_else(self, monkeypatch, tmp_path):
        from src.prepare_data.readers import gdb_reader, gml_reader, shp_reader, swde_reader

        monkeypatch.setattr(gdb_reader, "read_all_layers", lambda p: {"L": "from_gdb"})
        monkeypatch.setattr(gml_reader, "read_all_layers", lambda p: {"L": "from_gml"})
        monkeypatch.setattr(
            swde_reader, "read_all_layers", lambda p, crs="EPSG:2178": {"L": "from_swde"}
        )
        monkeypatch.setattr(shp_reader, "read_all_layers", lambda paths, m: {"L": "from_shp"})

        candidates = {
            "gdb": [tmp_path / "a.gdb"],
            "gml": [tmp_path / "a.gml"],
            "shp": [tmp_path / "a.shp"],
            "swde": [tmp_path / "a.swd"],
        }
        layers, fmt = _read_unit_layers(candidates, layer_name_map={})
        assert fmt == "gdb"
        assert layers == {"L": "from_gdb"}

    def test_falls_back_to_gml_when_gdb_empty(self, monkeypatch, tmp_path):
        from src.prepare_data.readers import gdb_reader, gml_reader, shp_reader, swde_reader

        monkeypatch.setattr(gdb_reader, "read_all_layers", lambda p: {})
        monkeypatch.setattr(gml_reader, "read_all_layers", lambda p: {"L": "from_gml"})
        monkeypatch.setattr(
            swde_reader, "read_all_layers", lambda p, crs="EPSG:2178": {"L": "from_swde"}
        )
        monkeypatch.setattr(shp_reader, "read_all_layers", lambda paths, m: {"L": "from_shp"})

        candidates = {
            "gdb": [tmp_path / "a.gdb"],
            "gml": [tmp_path / "a.gml"],
            "shp": [tmp_path / "a.shp"],
            "swde": [tmp_path / "a.swd"],
        }
        layers, fmt = _read_unit_layers(candidates, layer_name_map={})
        assert fmt == "gml"

    def test_falls_back_all_the_way_to_swde(self, monkeypatch, tmp_path):
        from src.prepare_data.readers import gdb_reader, gml_reader, shp_reader, swde_reader

        monkeypatch.setattr(gdb_reader, "read_all_layers", lambda p: {})
        monkeypatch.setattr(gml_reader, "read_all_layers", lambda p: {})
        monkeypatch.setattr(shp_reader, "read_all_layers", lambda paths, m: {})
        monkeypatch.setattr(
            swde_reader, "read_all_layers", lambda p, crs="EPSG:2178": {"L": "from_swde"}
        )

        candidates = {"gdb": [tmp_path / "a.gdb"], "swde": [tmp_path / "a.swd"]}
        layers, fmt = _read_unit_layers(candidates, layer_name_map={})
        assert fmt == "swde"

    def test_merges_layers_across_multiple_gdb_candidates(self, monkeypatch, tmp_path):
        # Real case confirmed on real data: some deliveries ship TWO complementary .gdb files
        # for the same unit -- one carries DzialkaEwidencyjna, a separate "..._swde.gdb"/
        # "..._kontury_uzytki.gdb" carries only KonturUzytkuGruntowego. Stopping at the first
        # non-empty candidate (old behavior) silently dropped whichever layer only the OTHER
        # file had.
        from src.prepare_data.readers import gdb_reader

        gdb_a = tmp_path / "a_z_GML.gdb"
        gdb_b = tmp_path / "b_kontury_uzytki_swde.gdb"

        def fake_read(path):
            if path == gdb_a:
                return {"EGB_DzialkaEwidencyjna": "dzialki"}
            if path == gdb_b:
                return {"G5G_UZG": "uzytki"}
            return {}

        monkeypatch.setattr(gdb_reader, "read_all_layers", fake_read)

        candidates = {"gdb": [gdb_a, gdb_b]}
        layers, fmt = _read_unit_layers(candidates, layer_name_map={})
        assert fmt == "gdb"
        assert layers == {"EGB_DzialkaEwidencyjna": "dzialki", "G5G_UZG": "uzytki"}

    def test_first_gdb_candidate_wins_on_layer_name_collision(self, monkeypatch, tmp_path):
        from src.prepare_data.readers import gdb_reader

        gdb_a = tmp_path / "a.gdb"
        gdb_b = tmp_path / "b.gdb"

        def fake_read(path):
            return {"L": "from_a" if path == gdb_a else "from_b"}

        monkeypatch.setattr(gdb_reader, "read_all_layers", fake_read)

        layers, fmt = _read_unit_layers({"gdb": [gdb_a, gdb_b]}, layer_name_map={})
        assert layers == {"L": "from_a"}

    def test_no_format_yields_anything_returns_none(self, monkeypatch, tmp_path):
        from src.prepare_data.readers import swde_reader

        monkeypatch.setattr(swde_reader, "read_all_layers", lambda p, crs="EPSG:2178": {})

        layers, fmt = _read_unit_layers({"swde": [tmp_path / "a.swd"]}, layer_name_map={})
        assert fmt is None
        assert layers == {}


class TestJewLayerName:
    def test_matches_legacy_g5g_jew_exactly(self):
        ds = _FakeDataSource(["G5ADR", "G5G_JEW", "G5G_DZE"])
        assert _jew_layer_name(ds) == "G5G_JEW"

    def test_matches_standard_egb_name(self):
        ds = _FakeDataSource(["EGB_Budynek", "EGB_JednostkaEwidencyjna"])
        assert _jew_layer_name(ds) == "EGB_JednostkaEwidencyjna"

    def test_matches_third_egb_suffix_convention_case_insensitive(self):
        ds = _FakeDataSource(
            [
                "powiat_dziadowski_2026_AB_egb_dzialkaewidencyjna",
                "powiat_dziadowski_2026_AB_egb_jednostkaewidencyjna",
            ]
        )
        assert _jew_layer_name(ds) == "powiat_dziadowski_2026_AB_egb_jednostkaewidencyjna"

    def test_returns_none_when_no_jew_layer_present(self):
        ds = _FakeDataSource(["G5ADR", "EGB_JednostkaRejestrowaGruntow"])
        assert _jew_layer_name(ds) is None


class TestCombinedPowiatPrefix:
    def test_detects_multi_gmina_delivery(self, monkeypatch, tmp_path):
        import src.prepare_data.extract_polygons as mod

        fake_ds = _FakeDataSource(["olsztynski_2026_egb_jednostkaewidencyjna"])
        monkeypatch.setattr(mod.ogr, "Open", lambda p, flag=0: fake_ds)
        fake_gdf = pd.DataFrame({"idjednostkiewid": ["281401_4", "281401_5", "281402_4"]})
        monkeypatch.setattr(mod.gpd, "read_file", lambda p, layer=None: fake_gdf)

        assert _combined_powiat_prefix(tmp_path / "powiat_olsztynski_2814.gdb") == "2814"

    def test_returns_none_for_single_gmina_file(self, monkeypatch, tmp_path):
        # A JEW layer with exactly one real gmina code is a normal, already-handled per-gmina
        # delivery (or an unrelated single-record file) -- never treated as combined, so a
        # correctly-matched-by-filename per-gmina .gdb is never double-counted via this path.
        import src.prepare_data.extract_polygons as mod

        fake_ds = _FakeDataSource(["G5G_JEW"])
        monkeypatch.setattr(mod.ogr, "Open", lambda p, flag=0: fake_ds)
        fake_gdf = pd.DataFrame({"G5IDJ": ["286201_1"]})
        monkeypatch.setattr(mod.gpd, "read_file", lambda p, layer=None: fake_gdf)

        assert _combined_powiat_prefix(tmp_path / "2862.gdb") is None

    def test_returns_none_when_file_cannot_open(self, monkeypatch, tmp_path):
        import src.prepare_data.extract_polygons as mod

        monkeypatch.setattr(mod.ogr, "Open", lambda p, flag=0: None)
        assert _combined_powiat_prefix(tmp_path / "broken.gdb") is None

    def test_returns_none_when_no_jew_layer(self, monkeypatch, tmp_path):
        import src.prepare_data.extract_polygons as mod

        fake_ds = _FakeDataSource(["G5ADR", "G5DOK"])
        monkeypatch.setattr(mod.ogr, "Open", lambda p, flag=0: fake_ds)
        assert _combined_powiat_prefix(tmp_path / "attrs_only.gdb") is None

    def test_ignores_invalid_codes(self, monkeypatch, tmp_path):
        # Real data confirmed: some rows in a JEW-like layer can be junk/placeholder values --
        # only entries actually matching the TERYT unit-code shape should count.
        import src.prepare_data.extract_polygons as mod

        fake_ds = _FakeDataSource(["EGB_JednostkaEwidencyjna"])
        monkeypatch.setattr(mod.ogr, "Open", lambda p, flag=0: fake_ds)
        fake_gdf = pd.DataFrame({"idJednostkiEwid": ["281401_4", None, "not-a-code"]})
        monkeypatch.setattr(mod.gpd, "read_file", lambda p, layer=None: fake_gdf)

        assert _combined_powiat_prefix(tmp_path / "x.gdb") is None

    def test_returns_none_when_codes_span_multiple_powiats(self, monkeypatch, tmp_path):
        # Real data confirmed: internal integration/staging files (e.g. under
        # "INTEGRACJA_BAZ_DANYCH_*" folders) can bundle the WHOLE VOIVODESHIP into one JEW
        # layer -- "Warminsko_Mazurskie_na_ATLAS_2021.gdb" carried 150 codes across all 21
        # powiat prefixes. Such a file is not a genuine single-powiat combined delivery and
        # must be rejected, not silently filed under whichever prefix happens to sort first.
        import src.prepare_data.extract_polygons as mod

        fake_ds = _FakeDataSource(["G5G_JEW"])
        monkeypatch.setattr(mod.ogr, "Open", lambda p, flag=0: fake_ds)
        fake_gdf = pd.DataFrame({"G5IDJ": ["280101_1", "280401_2", "281401_4"]})
        monkeypatch.setattr(mod.gpd, "read_file", lambda p, layer=None: fake_gdf)

        assert _combined_powiat_prefix(tmp_path / "Warminsko_Mazurskie_na_ATLAS_2021.gdb") is None


class TestDiscoverUnitsCombinedDelivery:
    def test_registers_combined_gdb_under_pseudo_unit_when_it_fills_a_gap(
        self, monkeypatch, tmp_path
    ):
        import src.prepare_data.extract_polygons as mod

        year_dir = tmp_path / "rok_2099"
        combined = year_dir / "powiat_test_2814.gdb"
        combined.mkdir(parents=True)

        monkeypatch.setattr(
            mod,
            "_combined_powiat_codes",
            lambda p: {"281401_4", "281402_2"} if p == combined else None,
        )

        units = mod._discover_units(year_dir)

        assert "2814_wspolny" in units
        assert units["2814_wspolny"]["gdb"] == [combined]

    def test_does_not_shadow_a_normally_matched_unit(self, monkeypatch, tmp_path):
        # A .gdb whose filename already matches UNIT_PATTERN must never be re-checked for the
        # combined-delivery path -- confirms the "continue" short-circuit in the discovery loop.
        import src.prepare_data.extract_polygons as mod

        year_dir = tmp_path / "rok_2099"
        (year_dir / "281701_1_szczytno.gdb").mkdir(parents=True)

        calls = []
        monkeypatch.setattr(mod, "_combined_powiat_codes", lambda p: calls.append(p) or None)

        units = mod._discover_units(year_dir)

        assert "281701_1" in units
        assert calls == []

    def test_skips_combined_delivery_when_all_gminy_already_covered_normally(
        self, monkeypatch, tmp_path
    ):
        # Real data confirmed: most powiats have BOTH a normal per-gmina delivery AND a
        # redundant whole-powiat staging copy (e.g. "*_dodane_atrybuty_swde.gdb") for the same
        # year. Registering the redundant copy would duplicate every already-covered gmina's
        # rows once merged into a shared parquet tree -- it must be skipped entirely when there
        # is no actual gap to fill.
        import src.prepare_data.extract_polygons as mod

        year_dir = tmp_path / "rok_2099"
        (year_dir / "281401_4_gm_a.gdb").mkdir(parents=True)
        (year_dir / "281402_2_gm_b.gdb").mkdir(parents=True)
        combined = year_dir / "powiat_test_dodane_atrybuty_swde.gdb"
        combined.mkdir(parents=True)

        monkeypatch.setattr(
            mod,
            "_combined_powiat_codes",
            lambda p: {"281401_4", "281402_2"} if p == combined else None,
        )

        units = mod._discover_units(year_dir)

        assert "2814_wspolny" not in units
        assert units["281401_4"]["gdb"] == [year_dir / "281401_4_gm_a.gdb"]
        assert units["281402_2"]["gdb"] == [year_dir / "281402_2_gm_b.gdb"]

    def test_skips_combined_gml_when_gminy_already_covered_via_gdb(self, monkeypatch, tmp_path):
        # Real bug confirmed on real data: nowomiejski 2024 already has full GDB coverage for
        # all 5 gminy, but a same-year combined *.gml* staging file was still registering as a
        # gap-filler, because a per-format check only looks at GML coverage (genuinely absent)
        # and ignores that GDB already covers it -- duplicating every gmina's rows once merged.
        # The gap check must be "does this gmina have ANY delivery already", not "this exact
        # container format".
        import src.prepare_data.extract_polygons as mod

        year_dir = tmp_path / "rok_2099"
        (year_dir / "281401_4_gm_a.gdb").mkdir(parents=True)
        (year_dir / "281402_2_gm_b.gdb").mkdir(parents=True)
        combined = year_dir / "powiat_test_dodane_atrybuty_swde.gml"
        combined.touch()

        monkeypatch.setattr(
            mod,
            "_combined_powiat_codes",
            lambda p: {"281401_4", "281402_2"} if p == combined else None,
        )

        units = mod._discover_units(year_dir)

        assert "2814_wspolny" not in units

    def test_registers_combined_delivery_when_only_some_gminy_are_covered(
        self, monkeypatch, tmp_path
    ):
        # Partial coverage: one gmina already has a normal delivery, another doesn't. The
        # combined file still gets registered as the pseudo-unit so the missing gmina's data
        # isn't lost -- accepted tradeoff (some duplicate rows for the already-covered gmina)
        # in exchange for not requiring a real per-row split.
        import src.prepare_data.extract_polygons as mod

        year_dir = tmp_path / "rok_2099"
        (year_dir / "281401_4_gm_a.gdb").mkdir(parents=True)
        combined = year_dir / "powiat_test_dodane_atrybuty_swde.gdb"
        combined.mkdir(parents=True)

        monkeypatch.setattr(
            mod,
            "_combined_powiat_codes",
            lambda p: {"281401_4", "281402_2"} if p == combined else None,
        )

        units = mod._discover_units(year_dir)

        assert "2814_wspolny" in units
        assert units["2814_wspolny"]["gdb"] == [combined]
