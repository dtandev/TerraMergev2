from pathlib import Path

from omegaconf import OmegaConf

from src.prepare_data.layers_merge import (
    _find_source_dirs,
    _matches_egb_suffix,
    _matches_year_suffixed_alias,
    run_layers_merge,
)

LAYER_NAME_MAP = {
    "DzialkaEwidencyjna": ["EGB_DzialkaEwidencyjna", "G5G_DZE"],
    "KonturUzytkuGruntowego": ["EGB_KonturUzytkuGruntowego", "G5G_UZG", "G5G_KKL"],
    "ObiektTrwaleZwiazanyZBudynkiem": [
        "EGB_ObiektTrwaleZwiazanyZBudynkiemPOLYGON"  # pragma: allowlist secret (cadastral layer name)
    ],
}


class TestMatchesEgbSuffix:
    def test_matches_plain_suffix(self):
        assert _matches_egb_suffix(
            "pow_bartoszycki_m_Bartoszyce_egb_dzialkaewidencyjna", "DzialkaEwidencyjna"
        )

    def test_matches_with_geometry_type_suffix(self):
        assert _matches_egb_suffix(
            "pow_braniewski_M_Braniewo_egb_obiekttrwalezwiazanyzbudynkiem_MultiPolygon",
            "ObiektTrwaleZwiazanyZBudynkiem",
        )

    def test_case_insensitive(self):
        assert _matches_egb_suffix("PREFIX_EGB_KonturUzytkuGruntowego", "KonturUzytkuGruntowego")

    def test_does_not_match_different_target(self):
        assert not _matches_egb_suffix(
            "pow_bartoszycki_m_Bartoszyce_egb_budynek", "DzialkaEwidencyjna"
        )

    def test_does_not_match_without_egb_marker(self):
        assert not _matches_egb_suffix("G5G_DZE", "DzialkaEwidencyjna")

    def test_matches_with_trailing_numeric_disambiguator(self):
        # Real data confirmed: rok_2026 elbląski delivery exports layers named
        # "elblski_egb_konturuzytkugruntowego_3" -- a GDAL-appended "_3" collision-avoidance
        # suffix, not part of the target name.
        assert _matches_egb_suffix("elblski_egb_konturuzytkugruntowego_3", "KonturUzytkuGruntowego")

    def test_does_not_match_plain_egb_target_alias(self):
        # exact aliases like "EGB_DzialkaEwidencyjna" have no "_egb_" marker (capital, no
        # underscore before "Ewidencyjna") -- suffix matching must not double-handle these,
        # exact-alias matching in _find_source_dirs already covers them.
        assert not _matches_egb_suffix("EGB_DzialkaEwidencyjna", "DzialkaEwidencyjna")


class TestMatchesYearSuffixedAlias:
    def test_matches_alias_with_trailing_year(self):
        # Real case confirmed: olecki rok_2025 delivery, folder
        # "EGB_KonturUzytkuGruntowego_2024" for a unit whose canonical alias carries no year.
        assert _matches_year_suffixed_alias(
            "EGB_KonturUzytkuGruntowego_2024", ["EGB_KonturUzytkuGruntowego", "G5G_UZG"]
        )

    def test_case_insensitive(self):
        assert _matches_year_suffixed_alias(
            "egb_dzialkaewidencyjna_2023", ["EGB_DzialkaEwidencyjna"]
        )

    def test_does_not_match_plain_alias_without_year_suffix(self):
        # Real crash confirmed: matching a plain alias here too (a no-op strip) causes the
        # SAME physical directory to be found twice on a case-insensitive filesystem (once via
        # the direct unit_dir/alias lookup, once via this check) -- run_layers_merge then tries
        # to move its contents twice, crashing with FileNotFoundError the second time.
        assert not _matches_year_suffixed_alias(
            "EGB_KonturUzytkuGruntowego", ["EGB_KonturUzytkuGruntowego"]
        )

    def test_does_not_match_unrelated_name(self):
        assert not _matches_year_suffixed_alias("EGB_Budynek_2024", ["EGB_KonturUzytkuGruntowego"])


class TestFindSourceDirsNoDuplicateEntries:
    def test_plain_alias_directory_found_exactly_once(self, tmp_path):
        # Real crash confirmed: on a case-insensitive filesystem, a lowercase-named directory
        # ("egb_budynek") is found both by the exact unit_dir/"EGB_Budynek" lookup AND (if the
        # year-suffix check wrongly allowed a no-op strip) by iterdir() -- two Path objects for
        # the SAME physical directory, so run_layers_merge tried to move its contents twice.
        unit_dir = tmp_path / "280101_1"
        (unit_dir / "EGB_Budynek").mkdir(parents=True)

        found = _find_source_dirs(unit_dir, "Budynek", ["EGB_Budynek", "EGB_BudynekPOLYGON"])

        assert len(found) == 1


def _write_parquet_stub(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")


class TestRunLayersMerge:
    def test_merges_suffix_named_layer_into_canonical_target(self, tmp_path):
        out_root = tmp_path / "parquets"
        unit_dir = out_root / "280101_1"
        src = unit_dir / "pow_bartoszycki_m_Bartoszyce_egb_dzialkaewidencyjna" / "year=2026"
        _write_parquet_stub(src / "input_data.parquet")

        cfg = OmegaConf.create(
            {
                "data": {"base_dir": str(tmp_path)},
                "prepare": {
                    "output_subdir": "parquets",
                    "merge": {"overwrite": False},
                    "layer_name_map": LAYER_NAME_MAP,
                },
            }
        )
        run_layers_merge(cfg)

        dest = unit_dir / "DzialkaEwidencyjna" / "year=2026" / "input_data.parquet"
        assert dest.exists()
        assert not src.exists()

    def test_still_merges_standard_exact_alias(self, tmp_path):
        out_root = tmp_path / "parquets"
        unit_dir = out_root / "280101_1"
        src = unit_dir / "G5G_DZE" / "year=2013"
        _write_parquet_stub(src / "input_data.parquet")

        cfg = OmegaConf.create(
            {
                "data": {"base_dir": str(tmp_path)},
                "prepare": {
                    "output_subdir": "parquets",
                    "merge": {"overwrite": False},
                    "layer_name_map": LAYER_NAME_MAP,
                },
            }
        )
        run_layers_merge(cfg)

        dest = unit_dir / "DzialkaEwidencyjna" / "year=2013" / "input_data.parquet"
        assert dest.exists()

    def test_suffix_and_exact_alias_both_present_are_both_merged(self, tmp_path):
        out_root = tmp_path / "parquets"
        unit_dir = out_root / "280101_1"
        src_exact = unit_dir / "G5G_KKL" / "year=2016"
        src_suffix = unit_dir / "some_prefix_egb_konturuzytkugruntowego" / "year=2017"
        _write_parquet_stub(src_exact / "a.parquet")
        _write_parquet_stub(src_suffix / "b.parquet")

        cfg = OmegaConf.create(
            {
                "data": {"base_dir": str(tmp_path)},
                "prepare": {
                    "output_subdir": "parquets",
                    "merge": {"overwrite": False},
                    "layer_name_map": LAYER_NAME_MAP,
                },
            }
        )
        run_layers_merge(cfg)

        target = unit_dir / "KonturUzytkuGruntowego"
        assert (target / "year=2016" / "a.parquet").exists()
        assert (target / "year=2017" / "b.parquet").exists()

    def test_merges_year_suffixed_exact_alias(self, tmp_path):
        # Real case confirmed: olecki rok_2025 delivery, folder
        # "EGB_KonturUzytkuGruntowego_2024" for a unit whose canonical alias carries no year.
        out_root = tmp_path / "parquets"
        unit_dir = out_root / "281304_5"
        src = unit_dir / "EGB_KonturUzytkuGruntowego_2024" / "year=2025"
        _write_parquet_stub(src / "input_data.parquet")

        cfg = OmegaConf.create(
            {
                "data": {"base_dir": str(tmp_path)},
                "prepare": {
                    "output_subdir": "parquets",
                    "merge": {"overwrite": False},
                    "layer_name_map": LAYER_NAME_MAP,
                },
            }
        )
        run_layers_merge(cfg)

        dest = unit_dir / "KonturUzytkuGruntowego" / "year=2025" / "input_data.parquet"
        assert dest.exists()
        assert not src.exists()
