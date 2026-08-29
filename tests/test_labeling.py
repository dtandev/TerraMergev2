import pandas as pd

from src.modeling.labeling import build_split_labels_full, build_uzg_conversion_labels


class TestBuildSplitLabelsFull:
    def test_detects_split_on_consecutive_years(self):
        df = pd.DataFrame(
            {
                "hex_id": ["h1", "h1", "h1"],
                "year": [2020, 2021, 2022],
                "shape_area_mean": [1000.0, 1000.0, 100.0],  # big drop only 2021 -> 2022
                "coverage_area": [500.0, 500.0, 500.0],  # conserved
                "n_parcel": [1, 1, 2],  # count grows on the drop year
            }
        )
        out = build_split_labels_full(
            df, n_parcels_col="n_parcel", area_conservation_tol=0.05, eps_abs=100.0
        )
        row_2022 = out[out["year"] == 2022].iloc[0]
        row_2021 = out[out["year"] == 2021].iloc[0]
        assert row_2022["split_proxy"] == True  # noqa: E712
        assert row_2021["y_next"] == True  # noqa: E712 (shift(-1) of the 2022 proxy)

    def test_year_gap_does_not_falsely_trigger_split(self):
        # Same "looks like a split" delta pattern as above, but across a missing year
        # (2020 -> 2022, no 2021 row) — must NOT be treated as a valid one-year transition.
        df = pd.DataFrame(
            {
                "hex_id": ["h2", "h2"],
                "year": [2020, 2022],
                "shape_area_mean": [1000.0, 100.0],
                "coverage_area": [500.0, 500.0],
                "n_parcel": [1, 2],
            }
        )
        out = build_split_labels_full(
            df, n_parcels_col="n_parcel", area_conservation_tol=0.05, eps_abs=100.0
        )
        row_2022 = out[out["year"] == 2022].iloc[0]
        assert row_2022["split_proxy"] == False  # noqa: E712

    def test_no_split_when_area_not_conserved(self):
        df = pd.DataFrame(
            {
                "hex_id": ["h3", "h3"],
                "year": [2020, 2021],
                "shape_area_mean": [1000.0, 100.0],
                "coverage_area": [500.0, 50.0],  # sum area collapsed too -> not conserved
                "n_parcel": [1, 2],
            }
        )
        out = build_split_labels_full(
            df, n_parcels_col="n_parcel", area_conservation_tol=0.05, eps_abs=100.0
        )
        row_2021 = out[out["year"] == 2021].iloc[0]
        assert row_2021["split_proxy"] == False  # noqa: E712


class TestBuildUzgConversionLabels:
    def test_detects_conversion_on_consecutive_years(self):
        df = pd.DataFrame(
            {
                "hex_id": ["h1", "h1"],
                "year": [2020, 2021],
                "uzg_R_share": [0.8, 0.2],
                "uzg_B_share": [0.1, 0.7],
                "sum_uzg": [1.0, 1.0],
            }
        )
        out = build_uzg_conversion_labels(
            df,
            share_R_col="uzg_R_share",
            share_B_col="uzg_B_share",
            sum_col="sum_uzg",
            area_conservation_tol=0.01,
        )
        row_2021 = out[out["year"] == 2021].iloc[0]
        assert row_2021["convert_proxy"] == True  # noqa: E712

    def test_year_gap_does_not_falsely_trigger_conversion(self):
        df = pd.DataFrame(
            {
                "hex_id": ["h2", "h2"],
                "year": [2020, 2023],  # 3-year gap
                "uzg_R_share": [0.8, 0.2],
                "uzg_B_share": [0.1, 0.7],
                "sum_uzg": [1.0, 1.0],
            }
        )
        out = build_uzg_conversion_labels(
            df,
            share_R_col="uzg_R_share",
            share_B_col="uzg_B_share",
            sum_col="sum_uzg",
            area_conservation_tol=0.01,
        )
        row_2023 = out[out["year"] == 2023].iloc[0]
        assert row_2023["convert_proxy"] == False  # noqa: E712
