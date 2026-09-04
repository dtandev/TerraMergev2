import pandas as pd
import pytest

from src.features.features_makeover import FeaturesMakeover

FM = FeaturesMakeover()


class TestUzgOzuSimple:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Bp", "B"),  # starts with B
            ("RIVa", "R"),  # starts with R
            ("Lz-RIVb", "L"),  # head before '-', starts with L
            ("W", "W"),
            ("Ps", "Ps"),  # starts with P
            ("dr", "dr"),  # exact lowercase special-case
            ("Ł", "Ł"),  # falls through to the exact set
        ],
    )
    def test_simplifies_codes(self, raw, expected):
        out = FM.add_uzg_ozu_simple(pd.DataFrame({"ozu": [raw]}))
        assert out["uzg_ozu_simple"].iloc[0] == expected

    def test_none_becomes_na(self):
        out = FM.add_uzg_ozu_simple(pd.DataFrame({"ozu": [None]}))
        assert pd.isna(out["uzg_ozu_simple"].iloc[0])

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            FM.add_uzg_ozu_simple(pd.DataFrame({"other": [1]}))


class TestUzgBonScore:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("IIIa", 4.0),  # III=4, 'A' no change
            ("III b", 3.5),  # space stripped, 'B' → -0.5
            ("ivB", 2.5),  # IV=3, 'B' → -0.5, case-insensitive
            ("VI", 1.0),
            ("I", 6.0),
        ],
    )
    def test_scores(self, raw, expected):
        out = FM.add_uzg_bon_score(pd.DataFrame({"ozk": [raw]}))
        assert float(out["uzg_bon_score"].iloc[0]) == expected

    def test_invalid_and_none_are_na(self):
        out = FM.add_uzg_bon_score(pd.DataFrame({"ozk": ["ZZ", None]}))
        assert out["uzg_bon_score"].isna().all()

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            FM.add_uzg_bon_score(pd.DataFrame({"other": [1]}))


class TestSanitizeMpzpSource:
    def test_commas_to_underscore_strip_and_fill(self):
        df = pd.DataFrame({"etykieta": [" MN,U ", None]})
        out = FeaturesMakeover._sanitize_mpzp_source(df, placeholder="Brak")
        assert out["etykieta"].iloc[0] == "MN_U"
        assert out["etykieta"].iloc[1] == "Brak"

    def test_writes_to_out_col_when_given(self):
        df = pd.DataFrame({"etykieta": ["MN"]})
        out = FeaturesMakeover._sanitize_mpzp_source(df, out_col="clean")
        assert out["clean"].iloc[0] == "MN"
        assert out["etykieta"].iloc[0] == "MN"  # source preserved

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            FeaturesMakeover._sanitize_mpzp_source(pd.DataFrame({"x": [1]}))


class TestNormalizeMpzpSymbol:
    def test_strips_plan_local_numbering_to_base_symbol(self):
        n = FeaturesMakeover._normalize_mpzp_symbol
        # leading plan/sheet prefix is dropped only when a symbol follows it
        assert n("10MN") == "MN"
        assert n("A102MN") == "MN"
        assert n("1-KDW") == "KDW"
        assert n("IV/ZL") == "ZL"
        # trailing variant number is dropped, but the symbol itself is kept (not eaten)
        assert n("ML1") == "ML"
        assert n("D3") == "D"
        assert n("MU4") == "MU"
        # road width codes and spaces collapse
        assert n("KD10/1X5/") == "KD"
        assert n("KD 10") == "KD"
        # clean symbols are unchanged (idempotent)
        assert n("R") == "R"
        assert n("MN-U") == "MN-U"


class TestAddMpzpLabelSimple:
    def test_maps_to_group_with_placeholder_fallback(self):
        df = pd.DataFrame({"etykieta": ["MN", "ZZZ"]})
        mapping = pd.DataFrame({"etykieta_oryginalna": ["MN"], "grupa_glowna": ["mieszkaniowa"]})
        out = FeaturesMakeover._add_mpzp_label_simple(df, mapping, placeholder="Brak")
        assert out["mpzp_etykieta"].tolist() == ["mieszkaniowa", "Brak"]

    def test_plan_local_labels_map_through_normalization(self):
        # A mapping keyed on the base symbol matches plan-local variants from any plan.
        df = pd.DataFrame({"etykieta": ["10MN", "A102MN", "5R", "ZZZ"]})
        mapping = pd.DataFrame({"etykieta_oryginalna": ["MN", "R"], "grupa_glowna": ["M", "R"]})
        out = FeaturesMakeover._add_mpzp_label_simple(df, mapping, placeholder="Brak")
        assert out["mpzp_etykieta"].tolist() == ["M", "M", "R", "Brak"]


class TestApplyMpzpTemporalRule:
    def test_future_plan_is_blanked(self):
        # plan enacted in 2022 but the data year is 2020 → the plan didn't exist yet.
        df = pd.DataFrame(
            {"mpzp_etykieta": ["mieszkaniowa"], "data_uchwaly": ["2022-06-01"], "year": [2020]}
        )
        out = FeaturesMakeover._apply_mpzp_temporal_rule(df, placeholder="Brak")
        assert out["mpzp_etykieta"].iloc[0] == "Brak"

    def test_past_plan_is_kept(self):
        df = pd.DataFrame(
            {"mpzp_etykieta": ["mieszkaniowa"], "data_uchwaly": ["2019-06-01"], "year": [2020]}
        )
        out = FeaturesMakeover._apply_mpzp_temporal_rule(df)
        assert out["mpzp_etykieta"].iloc[0] == "mieszkaniowa"

    def test_missing_date_left_untouched(self):
        df = pd.DataFrame(
            {"mpzp_etykieta": ["mieszkaniowa"], "data_uchwaly": [None], "year": [2020]}
        )
        out = FeaturesMakeover._apply_mpzp_temporal_rule(df)
        assert out["mpzp_etykieta"].iloc[0] == "mieszkaniowa"

    def test_missing_columns_return_frame_unchanged(self):
        df = pd.DataFrame({"mpzp_etykieta": ["x"]})  # no date/year cols
        out = FeaturesMakeover._apply_mpzp_temporal_rule(df)
        assert out["mpzp_etykieta"].iloc[0] == "x"
