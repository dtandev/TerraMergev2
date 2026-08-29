import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from src.features.add_uzg import (
    _append_early_kkl,
    _coalesce,
    _fill_ozk_strict,
    _norm_kkl,
    _norm_uzg,
)


def _square(x=0.0, y=0.0, s=1.0) -> Polygon:
    return Polygon([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])


class TestCoalesce:
    def test_first_non_na_wins_and_target_kept(self):
        # Realistic shape from _UZG_COALESCE: the target name is also the first source.
        df = pd.DataFrame({"OZU": [None, "R"], "G5OZU": ["Ł", "X"]})

        out = _coalesce(df, {"OZU": ["OZU", "G5OZU"]})

        assert list(out["OZU"]) == ["Ł", "R"]  # row0 falls back to G5OZU, row1 keeps OZU
        assert "OZU" in out.columns
        assert "G5OZU" not in out.columns  # consumed source dropped

    def test_target_created_from_alternatives_when_absent(self):
        df = pd.DataFrame({"G5IDT": ["a", None], "OTHER": [1, 2]})

        out = _coalesce(df, {"IDUZYTKU": ["IDUZYTKU", "G5IDT"]})

        assert list(out["IDUZYTKU"]) == ["a", None]
        assert "G5IDT" not in out.columns
        assert "OTHER" in out.columns  # untouched

    def test_missing_sources_are_a_noop(self):
        df = pd.DataFrame({"KEEP": [1]})
        out = _coalesce(df, {"OZU": ["G5OZU", "G5OFU"]})
        assert list(out.columns) == ["KEEP"]


class TestNormKkl:
    def _input(self, geom=None):
        return gpd.GeoDataFrame(
            {"G5OZU": ["R"], "G5OZK": ["RIVb"], "G5IDK": ["k1"], "ST_OBJ": ["v1"]},
            geometry=[geom or _square()],
            crs="EPSG:2180",
        )

    def test_renames_columns_to_canonical(self):
        out = _norm_kkl(self._input(), "EPSG:2180")
        assert {"OZU", "OZK", "IDKONTURU"} <= set(out.columns)
        assert not ({"G5OZU", "G5OZK", "G5IDK"} & set(out.columns))

    def test_wersjaid_backfilled_from_st_obj(self):
        out = _norm_kkl(self._input(), "EPSG:2180")
        assert out["WERSJAID"].iloc[0] == "v1"

    def test_string_columns_get_string_dtype(self):
        out = _norm_kkl(self._input(), "EPSG:2180")
        assert out["OZU"].dtype == "string"
        assert out["OZK"].dtype == "string"

    def test_invalid_geometry_is_repaired(self):
        bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])  # self-intersecting
        assert not bowtie.is_valid
        out = _norm_kkl(self._input(geom=bowtie), "EPSG:2180")
        assert out.geometry.iloc[0].is_valid

    def test_reprojects_to_target_crs(self):
        out = _norm_kkl(self._input(), "EPSG:2178")
        assert out.crs.to_epsg() == 2178


class TestNormUzg:
    def test_coalesces_ozu_and_iduzytku_then_renames(self):
        gdf = gpd.GeoDataFrame(
            {"G5OZU": ["Ł"], "G5IDT": ["u1"], "IDR": ["r1"]},
            geometry=[_square()],
            crs="EPSG:2180",
        )

        out = _norm_uzg(gdf, "EPSG:2180")

        assert out["OZU"].iloc[0] == "Ł"  # coalesced from G5OZU
        assert out["IDUZYTKU"].iloc[0] == "u1"  # coalesced from G5IDT
        assert out["IDENTIFIER"].iloc[0] == "r1"  # renamed from IDR
        assert out["OZU"].dtype == "string"


class TestAppendEarlyKkl:
    def _kug(self):
        return gpd.GeoDataFrame(
            {"IDUZYTKU": ["u1"], "OZU": ["R"], "year": [2020]},
            geometry=[_square()],
            crs="EPSG:2180",
        )

    def _kkl(self, years):
        n = len(years)
        return gpd.GeoDataFrame(
            {"IDKONTURU": [f"k{i}" for i in range(n)], "OZK": ["RIVb"] * n, "year": years},
            geometry=[_square(x=i) for i in range(n)],
            crs="EPSG:2180",
        )

    def test_appends_only_years_before_kug_min(self):
        out = _append_early_kkl(self._kug(), self._kkl([2015, 2021]))
        assert len(out) == 2  # only the 2015 KKL row is appended
        assert set(out["year"]) == {2020, 2015}

    def test_appended_row_takes_iduzytku_from_idkonturu(self):
        out = _append_early_kkl(self._kug(), self._kkl([2015]))
        appended = out[out["year"] == 2015].iloc[0]
        assert appended["IDUZYTKU"] == "k0"

    def test_no_early_kkl_returns_kug_unchanged(self):
        out = _append_early_kkl(self._kug(), self._kkl([2020, 2022]))
        assert len(out) == 1


class TestFillOzkStrict:
    def test_ozk_filled_by_containment_then_carried_forward(self):
        # Two UZG contours of the same parcel across years; only the 2018 one sits inside a
        # KKL contour that carries OZK. The spatial join fills 2018, then LOCF carries it to 2019.
        kug = gpd.GeoDataFrame(
            {"IDUZYTKU": ["u1", "u1"], "year": [2018, 2019], "OZK": [pd.NA, pd.NA]},
            geometry=[_square(0.1, 0.1, 0.2), _square(5, 5, 0.2)],
            crs="EPSG:2180",
        )
        kkl = gpd.GeoDataFrame(
            {"OZK": ["RIVb"], "year": [2018]},
            geometry=[_square(0, 0, 1)],  # contains the 2018 UZG square
            crs="EPSG:2180",
        )

        out = _fill_ozk_strict(kug, kkl).sort_values("year")

        assert list(out["OZK"]) == ["RIVb", "RIVb"]

    def test_no_containment_leaves_ozk_missing(self):
        kug = gpd.GeoDataFrame(
            {"IDUZYTKU": ["u1"], "year": [2019], "OZK": [pd.NA]},
            geometry=[_square(10, 10, 0.2)],
            crs="EPSG:2180",
        )
        kkl = gpd.GeoDataFrame(
            {"OZK": ["RIVb"], "year": [2018]},
            geometry=[_square(0, 0, 1)],
            crs="EPSG:2180",
        )

        out = _fill_ozk_strict(kug, kkl)

        assert out["OZK"].isna().all()
