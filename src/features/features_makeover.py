# features_makeover.py
from __future__ import annotations

import re

import numpy as np
import pandas as pd


class FeaturesMakeover:
    """
    Makijaże UZG:
      - uzg_ozu_simple: uproszczenie OZU
      - uzg_bon_score : bonitacja OZK wg reguły: I=6 ... VI=1; sufiks 'B' → -0.5
    """

    _BON_BASE = {"I": 6.0, "II": 5.0, "III": 4.0, "IV": 3.0, "V": 2.0, "VI": 1.0}
    # dopasowujemy CAŁOŚĆ napisu, kolejność od najdłuższych
    _BON_RE = re.compile(r"^(VI|IV|III|II|V|I)([ABZ])?$", re.IGNORECASE)

    def add_uzg_ozu_simple(
        self, df: pd.DataFrame, *, ozu_col: str = "ozu", out_col: str = "uzg_ozu_simple"
    ) -> pd.DataFrame:
        if ozu_col not in df.columns:
            raise KeyError(f"Brak kolumny '{ozu_col}' w DataFrame.")
        out = df.copy()
        out[out_col] = pd.Series(out[ozu_col], dtype="string").map(self._simplify_ozu)
        return out

    def add_uzg_bon_score(
        self, df: pd.DataFrame, *, ozk_col: str = "ozk", out_col: str = "uzg_bon_score"
    ) -> pd.DataFrame:
        if ozk_col not in df.columns:
            raise KeyError(f"Brak kolumny '{ozk_col}' w DataFrame.")
        out = df.copy()
        out[out_col] = (
            pd.Series(out[ozk_col], dtype="string").map(self._bon_score_rule).astype("Float64")
        )
        return out

    # --- prywatne ---
    @staticmethod
    def _simplify_ozu(value: str | None) -> str | None:
        if value is None or pd.isna(value):
            return np.nan
        head = str(value).strip().split("-", 1)[0]
        head_up = head.upper()
        if head == "dr":
            return "dr"
        if head_up.startswith("B"):
            return "B"
        if head_up.startswith("L"):
            return "L"
        if head_up.startswith("W"):
            return "W"
        if head_up.startswith("T"):
            return "T"
        if head_up.startswith("K"):
            return "K"
        if head_up.startswith("P"):
            return "Ps"
        if head_up.startswith("R"):
            return "R"
        if head in ("R", "Ł", "Ps", "S"):
            return head
        return head_up

    def _bon_score_rule(self, s: str | None) -> float:
        """
        I=6, II=5, III=4, IV=3, V=2, VI=1; sufiks 'B' → -0.5; 'A' bez zmiany.
        Przykłady: 'IIIa'→4.0, 'III b'→3.5, 'ivB'→2.5, 'VI'→1.0.
        """
        if not isinstance(s, str) or not s.strip():
            return np.nan
        # ujednolicenie zapisu
        txt = s.strip().upper().replace(" ", "")
        m = self._BON_RE.fullmatch(txt)  # dopasuj cały ciąg
        if not m:
            return np.nan
        base = self._BON_BASE[m.group(1)]
        sub = m.group(2)
        if sub == "B":
            base -= 0.5
        return float(base)

    # ─────────── 1) Sanitizacja źródła ───────────
    @staticmethod
    def _sanitize_mpzp_source(
        df: pd.DataFrame,
        *,
        src_col: str = "etykieta",
        placeholder: str = "Brak",
        sanitize_commas: bool = True,
        out_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Czyści kolumnę z etykietą: przecinki→'_', strip, uzupełnia NaN → placeholder.
        Jeśli podasz out_col, zapisze wynik do nowej kolumny; inaczej nadpisze src_col.
        """
        if src_col not in df.columns:
            raise KeyError(f"Brak kolumny '{src_col}' w DataFrame.")
        out = df.copy()
        col = out_col or src_col
        ser = out[src_col].astype("string")
        if sanitize_commas:
            ser = ser.str.replace(",", "_", regex=False)
        ser = ser.str.strip().fillna(placeholder)
        out[col] = ser
        return out

    # ─────────── 2) Normalizacja etykiety planu do symbolu bazowego ───────────
    @staticmethod
    def _normalize_mpzp_symbol(value: object) -> str:
        """Sprowadź etykietę MPZP do symbolu bazowego wspólnego dla różnych planów.

        Etykieta z WFS niesie numerację LOKALNĄ dla planu ("10MN", "A102MN", "1-KDW",
        "IV/R"), często z kodem szerokości drogi ("KD10/1X5/", "D12(1X6)"). Ten numer to
        indeks wielokąta w konkretnym planie — nie przenosi się między powiatami, więc
        dopasowanie po pełnej etykiecie daje nikłe pokrycie. Zdejmujemy prefiks/sufiks
        planowy i zostawiamy sam symbol przeznaczenia (MN, R, ZL, KDW, ...), który jest
        standardowy i mapuje się jednakowo wszędzie.
        """
        s = "" if value is None else str(value)
        s = s.strip().upper().replace(",", "_")
        if not s:
            return ""
        s = re.sub(r"/[^/]*/", "", s)  # kod szerokości drogi: KD10/1X5/
        s = re.sub(r"\([^)]*\)", "", s)  # wariant w nawiasie: (1X5)
        s = re.sub(r"\s+", "", s)  # scal spacje: "KD 10" → "KD10"
        s = re.sub(r"^[IVX]+/", "", s)  # rzymski prefiks planu: IV/
        # Plan-lokalny prefiks (numer arkusza/wielokąta): A102MN, 10MN, 1-KDW. Zdejmujemy TYLKO
        # gdy po numerze idzie litera — inaczej "ML1"/"D3" (symbol + numer wariantu) zostałyby
        # zjedzone do pustego. Numer-wariant na końcu (MU4 → MU) zdejmuje osobna reguła niżej.
        s = re.sub(r"^[A-Z]{0,2}\d+[-_.\s]?(?=[A-Z])", "", s)
        s = re.sub(r"\d+$", "", s)  # końcowy numer wariantu: MU4 → MU, ML1 → ML
        return s.strip("-_. ")

    # ─────────── 3) Mapowanie do grupy głównej ───────────
    @staticmethod
    def _add_mpzp_label_simple(
        df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        *,
        src_col: str = "etykieta",
        out_col: str = "mpzp_etykieta",
        mapping_orig_col: str = "etykieta_oryginalna",
        mapping_group_col: str = "grupa_glowna",
        placeholder: str = "Brak",
    ) -> pd.DataFrame:
        """Znormalizuj etykietę do symbolu → zmapuj do grupy → (opcjonalnie) reguła czasowa.

        Zarówno źródłowa etykieta, jak i klucze mapowania przechodzą przez
        `_normalize_mpzp_symbol`, więc mapowanie zbudowane na jednym powiecie działa na
        innych, a plik mapujący może trzymać albo pełne etykiety, albo już same symbole.
        """
        out = df.copy()

        # 1) etykieta źródłowa → symbol bazowy (NaN → placeholder)
        ser = out[src_col].map(FeaturesMakeover._normalize_mpzp_symbol)
        ser = ser.mask(ser.eq(""), placeholder)

        # 2) mapowanie: symbol(etykieta_oryginalna) -> grupa_glowna (klucze też normalizowane)
        mp = mapping_df.copy()
        mp[mapping_orig_col] = mp[mapping_orig_col].map(FeaturesMakeover._normalize_mpzp_symbol)
        mapping = mp.drop_duplicates(mapping_orig_col).set_index(mapping_orig_col)[
            mapping_group_col
        ]

        # 3) zmapuj do grupy głównej z fallbackiem
        out[out_col] = ser.map(mapping).fillna(placeholder)

        return out

    # ─────────── 3) Reguła czasowa ───────────
    @staticmethod
    def _apply_mpzp_temporal_rule(
        df: pd.DataFrame,
        *,
        out_col: str = "mpzp_etykieta",
        plan_date_col: str = "data_uchwaly",
        year_col: str = "year",
        placeholder: str = "Brak",
    ) -> pd.DataFrame:
        """
        Jeśli data_uchwaly_plan.year > year → out_col = placeholder.
        (Nie rusza wierszy z brakami daty/roku.)
        """
        need = {out_col, plan_date_col, year_col}
        missing = need - set(df.columns)
        if missing:
            # jeśli brak kolumn, zwracamy bez zmian (bez dramatu)
            return df.copy()

        out = df.copy()
        plan_dt = pd.to_datetime(out[plan_date_col], errors="coerce")
        year_num = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
        mask = plan_dt.notna() & year_num.notna() & (plan_dt.dt.year > year_num)
        out.loc[mask, out_col] = placeholder
        return out
