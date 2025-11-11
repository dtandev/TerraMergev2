from __future__ import annotations
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from src.common.io_utils import setup_logging

from src.prepare_data.clean_directories import clean_directories
from src.prepare_data.extract_polygons_from_gdb import run_extraction_polygons
from src.prepare_data.layers_merge import run_layers_merge
from src.prepare_data.duckdb_init import run_duckdb_init
from src.prepare_data.clean_dataset import run_clean_dataset
from src.features.add_uzg import run_add_uzg

from src.features.add_transaction_prices import run_load_transactions
from src.features.add_geometric_features import run_add_geometric_features


def _sel(cfg: DictConfig, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, key, default=default)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_all(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    log_dir = Path(_sel(cfg, "logging.log_dir", "logs")).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        log_dir=log_dir,
        run_name="rule_them_all",
        console_level=str(_sel(cfg, "logging.console_level", "INFO")),
        file_level=str(_sel(cfg, "logging.file_level", "DEBUG")),
        fmt=str(_sel(cfg, "logging.format", "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")),
    )

    logger.info("Hydra working dir: {}", Path.cwd())

    base_dir_str = _sel(cfg, "data.base_dir")

    if not base_dir_str:
        logger.error("Missing required config key: data.base_dir")
        return
    base_dir = Path(base_dir_str).expanduser().resolve()

    # === DUCKDB INIT ===
    if _sel(cfg, "duckdb.init", False):
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

    if _sel(cfg, "prepare.enabled", False):
        logger.info("STEP[prepare_data] Start")

        # === CLEAN DIR ===
        if _sel(cfg, "prepare.clean.enabled", False):
            remove_dir_names = list(_sel(cfg, "prepare.clean.remove_dir_names", []))
            logger.info("STEP[cleanup] Start | base_dir={} | targets={}", base_dir, remove_dir_names)
            try:
                clean_directories(base_dir=base_dir, remove_dir_names=remove_dir_names)
                logger.success("STEP[cleanup] Done")
            except Exception:
                logger.exception("STEP[cleanup] Failed")
                raise
        else:
            logger.info("STEP[cleanup] Skipped (disabled)")

        # === PREPARE_DATA ===
        if _sel(cfg, "prepare.extract.enabled", False):
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
        if _sel(cfg, "prepare.merge.enabled", False):
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
        if _sel(cfg, "prepare.clean_dataset.enabled", False):
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

    if _sel(cfg, "features.enabled", False):
        logger.info("FEATURE ENGINEERING steps starting")

        # === ADD UZG ===
        if _sel(cfg, "features.add_uzg.enabled", False):
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
        if _sel(cfg, "features.add_transaction_prices.enabled", False):
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
        if _sel(cfg, "features.add_mpzp.enabled", False):
            logger.info("STEP[add_mpzp] Start")
            try:
                if _sel(cfg, "features.add_mpzp.for_parcels", False):
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
        if _sel(cfg, "features.add_geometric_features.enabled", False):

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


    # MAKE HEXAGONS 
    if _sel(cfg, "pipeline.make", _sel(cfg, "make", False)):
        logger.info("FEATURE ENGINEERING steps starting")
        if _sel(cfg, "pipeline.make_hexagons.enabled", _sel(cfg, "make_hexagons.enabled", False)):
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


        if _sel(cfg, "pipeline.add_parcels_data.enabled", False):
            logger.info("STEP[add_parcels_data_hex] Start")
            try:
                from v2.src.features.add_parcels_data_hexs import run_add_parcels_data      
                logger.info(
                    "Filling hexagons with %s data",
                    _sel(cfg, "add_parcels_data.join_with", "")
                )
                run_add_parcels_data(cfg)
                logger.success("STEP[add_parcels_data_hex] Done")
            except Exception:
                logger.exception("STEP[add_parcels_data_hex] Failed")
                raise
        else:
            logger.info("STEP[add_parcels_data_hex] Skipped (disabled)")



        if _sel(cfg, "pipeline.add_transactions_data.enabled", False):
            logger.info("STEP[transactions_hex] Start")
            try:
                from src.features.add_transactions_hex import run_add_transactions_hex      
                logger.info(
                    f"Filling hexagons with {_sel(cfg, 'pipeline.add_transactions_data.join_with', '')} data"
                )
                run_add_transactions_hex(cfg)
                logger.success("STEP[transactions_hex] Done")
            except Exception:
                logger.exception("STEP[transactions_hex] Failed")
                raise
        else:
            logger.info("STEP[transactions_hex] Skipped (disabled)")


        if _sel(cfg, "pipeline.add_mpzp_data.enabled", False):
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


        if _sel(cfg, "pipeline.add_kug_data.enabled", False):
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
    if _sel(cfg, "dataset.enabled", False):
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
    if _sel(cfg, "model.enabled", False):
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
    run_all()
