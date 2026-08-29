from __future__ import annotations

import sys
from pathlib import Path

# Must be imported before any `osgeo` (GDAL) import anywhere in the process — importing GDAL's
# bindings first and pyarrow.dataset afterwards segfaults at interpreter shutdown in this
# environment (verified: reversing the order avoids it). extract_polygons below imports
# osgeo.ogr, and add_uzg imports pyarrow.dataset, so without this line the import order would be
# wrong by accident of module load order.
import pyarrow.dataset  # noqa: F401
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from src.common.config_loader import load_config
from src.common.config_utils import sel
from src.common.io_utils import setup_logging
from src.features.add_geometric_features import run_add_geometric_features
from src.features.add_transaction_prices import run_load_transactions
from src.features.add_uzg import run_add_uzg
from src.prepare_data.clean_dataset import run_clean_dataset
from src.prepare_data.clean_directories import clean_directories
from src.prepare_data.duckdb_init import run_duckdb_init
from src.prepare_data.extract_polygons import run_extraction_polygons
from src.prepare_data.layers_merge import run_layers_merge

# Loaded before the config is composed in __main__ below — conf/config.yaml references
# variables via ${oc.env:VAR_NAME}, which OmegaConf reads from os.environ at resolve time.
load_dotenv()


def run_all(cfg: DictConfig) -> None:
    log_dir = Path(sel(cfg, "logging.log_dir", "logs")).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        log_dir=log_dir,
        run_name="terramerge",
        console_level=str(sel(cfg, "logging.console_level", "INFO")),
        file_level=str(sel(cfg, "logging.file_level", "DEBUG")),
        fmt=str(
            sel(
                cfg,
                "logging.format",
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
            )
        ),
    )

    logger.info("Working dir: {}", Path.cwd())

    base_dir_str = sel(cfg, "data.base_dir")

    if not base_dir_str:
        logger.error("Missing required config key: data.base_dir")
        return
    base_dir = Path(base_dir_str).expanduser().resolve()

    # === DUCKDB INIT ===
    if sel(cfg, "duckdb.init", False):
        logger.info("STEP[duckdb_init] Start")
        try:
            run_duckdb_init(cfg)
            logger.success("STEP[duckdb_init] Done")
        except Exception:
            logger.exception("STEP[duckdb_init] Failed")
            raise
    else:
        logger.info("STEP[duckdb_init] Skipped (disabled)")

    # ----------------------------- #
    # === PREPARE DATA PIPELINE === #
    # ----------------------------- #

    if sel(cfg, "prepare.enabled", False):
        logger.info("STEP[prepare_data] Start")

        # === CLEAN DIR ===
        if sel(cfg, "prepare.clean.enabled", False):
            remove_dir_names = list(sel(cfg, "prepare.clean.remove_dir_names", []))
            logger.info(
                "STEP[cleanup] Start | base_dir={} | targets={}", base_dir, remove_dir_names
            )
            try:
                clean_directories(base_dir=base_dir, remove_dir_names=remove_dir_names)
                logger.success("STEP[cleanup] Done")
            except Exception:
                logger.exception("STEP[cleanup] Failed")
                raise
        else:
            logger.info("STEP[cleanup] Skipped (disabled)")

        # === PREPARE_DATA ===
        if sel(cfg, "prepare.extract.enabled", False):
            logger.info("STEP[prepare_data] Start")
            try:
                run_extraction_polygons(cfg)
                logger.success("STEP[prepare_data] Done")
            except Exception:
                logger.exception("STEP[prepare_data] Failed")
                raise
        else:
            logger.info("STEP[prepare_data] Skipped (disabled)")

        # === LAYERS_MERGE ===
        if sel(cfg, "prepare.merge.enabled", False):
            logger.info("STEP[layers_merge] Start")
            try:
                run_layers_merge(cfg)
                logger.success("STEP[layers_merge] Done")
            except Exception:
                logger.exception("STEP[layers_merge] Failed")
                raise
        else:
            logger.info("STEP[layers_merge] Skipped (disabled)")

        # === CLEAN DATASET ===
        if sel(cfg, "prepare.clean_dataset.enabled", False):
            logger.info("STEP[clean_dataset] Start")
            try:
                run_clean_dataset(cfg)
                logger.success("STEP[clean_dataset] Done")
            except Exception:
                logger.exception("STEP[clean_dataset] Failed")
                raise
        else:
            logger.info("STEP[clean_dataset] Skipped (disabled)")

    else:
        logger.info("PREPARE_DATA steps skipped (disabled)")

    # --------------------------- #
    # === FEATURE ENGINEERING === #
    # --------------------------- #

    if sel(cfg, "features.enabled", False):
        logger.info("FEATURE ENGINEERING steps starting")

        # === ADD UZG ===
        if sel(cfg, "features.add_uzg.enabled", False):
            logger.info("STEP[add_uzg] Start")
            try:
                run_add_uzg(cfg)
                logger.success("STEP[add_uzg] Done")
            except Exception:
                logger.exception("STEP[add_uzg] Failed")
                raise
        else:
            logger.info("STEP[add_uzg] Skipped (disabled)")

        # === ADD_TRANSACTION_PRICES ===
        if sel(cfg, "features.add_transaction_prices.enabled", False):
            logger.info("STEP[add_transaction_prices] Start")
            try:
                run_load_transactions(cfg)
                logger.success("STEP[add_transaction_prices] Done")
            except Exception:
                logger.exception("STEP[add_transaction_prices] Failed")
                raise
        else:
            logger.info("STEP[add_transaction_prices] Skipped (disabled)")

        # === ADD_MPZP ===
        if sel(cfg, "features.add_mpzp.enabled", False):
            logger.info("STEP[add_mpzp] Start")
            try:
                if sel(cfg, "features.add_mpzp.for_parcels", False):
                    logger.info("Adding MPZP info linked to parcels")
                    from src.features.add_mpzp_for_parcels import run_add_mpzp
                else:
                    logger.info("Adding general MPZP info (not linked to parcels)")
                    from src.features.add_mpzp import run_add_mpzp

                run_add_mpzp(cfg)
                logger.success("STEP[add_mpzp] Done")
            except Exception:
                logger.exception("STEP[add_mpzp] Failed")
                raise
        else:
            logger.info("STEP[add_mpzp] Skipped (disabled)")

        # === ADD_GEOMETRIC_FEATURES ===
        if sel(cfg, "features.add_geometric_features.enabled", False):
            logger.info("STEP[add_geometric_features] Start")
            try:
                run_add_geometric_features(cfg)
                logger.success("STEP[add_geometric_features] Done")
            except Exception:
                logger.exception("STEP[add_geometric_features] Failed")
                raise
        else:
            logger.info("STEP[add_geometric_features] Skipped (disabled)")

    else:
        logger.info("FEATURE ENGINEERING steps skipped (disabled)")

    # ------------------------ #
    # === MAKE HEXAGONS    === #
    # ------------------------ #

    if sel(cfg, "pipeline.make", False):
        logger.info("HEXAGON BUILD steps starting")

        if sel(cfg, "pipeline.make_hexagons.enabled", False):
            logger.info("STEP[make_hexagons] Start")
            try:
                from src.features.make_empty_hexs import run_make_hexagons

                run_make_hexagons(cfg)
                logger.success("STEP[make_hexagons] Done")
            except Exception:
                logger.exception("STEP[make_hexagons] Failed")
                raise
        else:
            logger.info("STEP[make_hexagons] Skipped (disabled)")

        if sel(cfg, "pipeline.add_parcels_data.enabled", False):
            logger.info("STEP[add_parcels_data_hex] Start")
            try:
                from src.features.add_parcels_data_hexs import run_add_parcels_data

                logger.info(
                    "Filling hexagons with {} data",
                    sel(cfg, "pipeline.add_parcels_data.join_with", ""),
                )
                run_add_parcels_data(cfg)
                logger.success("STEP[add_parcels_data_hex] Done")
            except Exception:
                logger.exception("STEP[add_parcels_data_hex] Failed")
                raise
        else:
            logger.info("STEP[add_parcels_data_hex] Skipped (disabled)")

        if sel(cfg, "pipeline.add_transactions_data.enabled", False):
            logger.info("STEP[transactions_hex] Start")
            try:
                from src.features.add_transactions_hex import run_add_transactions_hex

                logger.info(
                    "Filling hexagons with {} data",
                    sel(cfg, "pipeline.add_transactions_data.join_with", ""),
                )
                run_add_transactions_hex(cfg)
                logger.success("STEP[transactions_hex] Done")
            except Exception:
                logger.exception("STEP[transactions_hex] Failed")
                raise
        else:
            logger.info("STEP[transactions_hex] Skipped (disabled)")

        if sel(cfg, "pipeline.add_mpzp_data.enabled", False):
            logger.info("STEP[add_mpzp_data] Start")
            try:
                from src.features.add_mpzp_hexs import run_add_mpzp_hexs

                run_add_mpzp_hexs(cfg)
                logger.success("STEP[add_mpzp_data] Done")
            except Exception:
                logger.exception("STEP[add_mpzp_data] Failed")
                raise
        else:
            logger.info("STEP[add_mpzp_data] Skipped (disabled)")

        if sel(cfg, "pipeline.add_kug_data.enabled", False):
            logger.info("STEP[add_kug_data] Start")
            try:
                from src.features.add_uzg_hexs import run_add_kug_hexs

                run_add_kug_hexs(cfg)
                logger.success("STEP[add_kug_data] Done")
            except Exception:
                logger.exception("STEP[add_kug_data] Failed")
                raise
        else:
            logger.info("STEP[add_kug_data] Skipped (disabled)")

    else:
        logger.info("MAKE HEXAGONS steps skipped (disabled)")

    # PREPARE LABELS
    if sel(cfg, "dataset.enabled", False):
        logger.info("STEP[prepare labels] Start")
        try:
            from src.modeling.labeling import run_creating_labels

            run_creating_labels(cfg)
        except Exception:
            logger.exception("STEP[prepare labels] Failed")
            raise

        try:
            from src.modeling.neighborhood import run_compute_neighbor_aggregates

            run_compute_neighbor_aggregates(cfg)
            logger.success("STEP[neighbor aggregation] Done")
        except Exception:
            logger.exception("STEP[neighbor aggregation] Failed")
            raise
    else:
        logger.info("STEP[prepare labels] Skipped (disabled)")

    # MODEL TRAINING
    if sel(cfg, "model.enabled", False):
        logger.info("STEP[model training] Start")
        try:
            from src.modeling.train_model import run_training

            run_training(cfg)
            logger.success("STEP[model training] Done")
        except Exception:
            logger.exception("STEP[model training] Failed")
            raise
    else:
        logger.info("STEP[model training] Skipped (disabled)")

    logger.info("All steps completed.")


if __name__ == "__main__":
    # CLI overrides use Hydra-style dot paths, e.g.
    #   uv run python -m src.main data.base_dir=/data/egib prepare.enabled=true
    run_all(load_config(sys.argv[1:]))
