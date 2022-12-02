import numbers
import csv
import sys
from tqdm import tqdm
import numpy as np
import warnings
import pandas
import os
from pathlib import Path
import json
import wandb
import logging
import argparse
import datetime
from typing import Any, Optional, Union

from bitorch.models import model_from_name
from bitorch.models.base import Model
from bitorch.models.model_hub import download_version_table, convert_dtypes

from examples.image_classification.datasets import dataset_from_name
from examples.image_classification.utils.utils import configure_logging

# from examples.image_classification.utils.arg_parser import create_argparser
from importlib import import_module

warnings.filterwarnings("ignore")
INFINITY = 1e4


def add_missing_columns(*key_lists: list, init_values: Any = None, table: pandas.DataFrame) -> pandas.DataFrame:
    for list_idx, key_list in enumerate(key_lists):
        for key_idx, key in enumerate(key_list):
            if key not in table.columns:
                value = (
                    init_values
                    if not isinstance(init_values, list)
                    else (
                        init_values[list_idx]
                        if not isinstance(init_values[list_idx], list)
                        else init_values[list_idx][key_idx]
                    )
                )
                table[key] = value
    return table


def extract_model_parameters(run: Any) -> dict:
    if "model_config" in run.config.keys():
        model_kwargs = run.config["model_config"]
    else:
        logging.info(
            "DEPRECATED: No model_config entry in run config found! Trying to reconstruct model parameters by parsing intial run command (experimental)..."
        )

        logging.debug("downloading run metadata...")
        run.file("wandb-metadata.json").download("/tmp", replace=True)

        metadata_path = Path("/tmp/wandb-metadata.json")
        if not metadata_path.exists():
            logging.error("metadata file could not be downloaded! skipping run...")
            return {}

        with metadata_path.open() as metadata_file:
            metadata = json.load(metadata_file)

        if "dlrm" in metadata["program"]:
            if "examples/dlrm" not in sys.path:
                sys.path.append("examples/dlrm")
            create_argparser = import_module("examples.dlrm.utils.arg_parser").create_argparser
        else:
            if "examples/image_classification" not in sys.path:
                sys.path.append("examples/image_classification")
            create_argparser = import_module("examples.image_classification.utils.arg_parser").create_argparser

        parser, model_parser = create_argparser(metadata["args"])
        parser.exit_on_error = False
        model_parser.exit_on_error = False

        args_, unparsed_model_args = parser.parse_known_args(metadata["args"])
        model_args_, _ = model_parser.parse_known_args(unparsed_model_args)

        model_kwargs = vars(model_args_)
        if not "dlrm" in metadata["program"]:
            dataset = dataset_from_name(args_.dataset)
            model_kwargs["input_shape"] = dataset.shape
            model_kwargs["num_classes"] = dataset.num_classes
            model_kwargs["model_name"] = args_.model
        else:
            model_kwargs["model_name"] = "dlrm"
        logging.debug(f"extracted model config: {model_kwargs}")
    return model_kwargs


def extract_artifact_version(artifact: wandb.Artifact, model_name: str) -> int:
    for alias in artifact._attrs["aliases"]:
        if (
            alias["artifactCollectionName"] == model_name
            and alias["alias"][0] == "v"
            and alias["alias"][1:].isnumeric()
        ):
            return int(alias["alias"][1:])
    return 0


def upload_model_to_hub(run: Any, model_name: str, api: wandb.Api) -> int:
    model_hub_path = f"{model_from_name(model_name).model_hub_base_path}/{model_name}"
    run_artifact = run.logged_artifacts()[0]
    run_artifact.link(model_hub_path)
    uploaded_artifact = api.artifact(f"{model_hub_path}:latest")
    return extract_artifact_version(uploaded_artifact, model_name)


def compare(configA: dict, configB: dict, compare_metrics: list) -> bool:
    for metric_name, mode in compare_metrics:
        if metric_name not in configA or metric_name not in configB or configA[metric_name] == configB[metric_name]:
            continue
        # this should be correct...
        return (configA[metric_name] < configB[metric_name]) == (mode == "max")
    return False


def add_model_to_version_table(
    model_kwargs: dict, version_table: pandas.DataFrame, run: Any, api: wandb.Api
) -> pandas.DataFrame:
    model_kwargs["model_hub_version"] = upload_model_to_hub(run, model_kwargs["model_name"], api)
    df_to_add = pandas.DataFrame([model_kwargs])
    return pandas.concat([version_table, df_to_add], ignore_index=True)


def update_table(
    version_table: pandas.DataFrame, model_kwargs: dict, run: Any, compare_metrics: list, api: wandb.Api
) -> pandas.DataFrame:
    model_comparison_keys = list(model_kwargs.keys())
    model_kwargs = convert_dtypes(model_kwargs)
    version_table = add_missing_columns(
        model_kwargs.keys(),  # type: ignore
        dict(run.summary).keys(),  # type: ignore
        [metric_name for metric_name, _ in compare_metrics],
        ["time uploaded", "model_hub_version"],
        init_values=[
            None,
            None,
            [-INFINITY if mode == "min" else +INFINITY for mode in [mode for _, mode in compare_metrics]],
            ["", "latest"],
        ],
        table=version_table,
    )

    logging.info(f"extracted model config: {model_kwargs}")
    model_kwargs_series = pandas.Series(model_kwargs)
    existing_row = version_table[(version_table[model_comparison_keys] == model_kwargs_series).all(1)]

    model_kwargs.update(dict(run.summary))
    model_kwargs["time uploaded"] = str(datetime.datetime.now())

    if existing_row.empty:
        logging.info("adding new model configuration to version table...")
        version_table = add_model_to_version_table(model_kwargs, version_table, run, api)
    else:
        existing_row_idx = np.where((version_table[model_comparison_keys] == model_kwargs_series).all(1))[0][0]
        existing_model_kwargs = version_table.iloc[existing_row_idx].to_dict()

        # this prevents reuploading the same model by favoring the older model if the other metrics are the same
        compare_metrics.append(["time uploaded", "min"])
        new_model_better = compare(existing_model_kwargs, model_kwargs, compare_metrics)
        if new_model_better:
            logging.info("overwriting preexisting model configuration in version table with better version...")
            version_table = version_table.drop(existing_row_idx).reset_index(drop=True)
            version_table = add_model_to_version_table(model_kwargs, version_table, run, api)
        else:
            logging.info("better/older model with same config already exists in version table, skipping...")
    return version_table


def write_table(version_table: pandas.DataFrame, model_name: str) -> None:
    version_table.to_csv(f"/tmp/versions.csv", quoting=csv.QUOTE_NONNUMERIC, index=False)
    entity, project, _ = model_from_name(model_name).version_table_path.split("/")
    with wandb.init(entity=entity, project=project) as run:  # type: ignore
        version_table_artifact = wandb.Artifact("model-tables", type="tables")
        version_table_artifact.add_file("/tmp/versions.csv")
        run.log_artifact(version_table_artifact)
        run.link_artifact(version_table_artifact, model_from_name(model_name).version_table_path)


def parse_metrics(metrics: list) -> list:
    compare_metrics = []
    for metric in metrics:
        parts = metric.split("/")
        metric_name = "/".join(parts[:-1])
        mode = parts[-1]
        if metric_name is None or mode is None or mode not in ["min", "max"]:
            logging.error(
                f"metric cannot be parsed: {metric}. Needs to be in format <metric name>/<mode: min or max>! e.g.: accuracy/max"
            )
            continue
        compare_metrics.append([metric_name, mode])
    return compare_metrics


def delete_model_version_table_in_hub() -> None:
    logging.warn("DELETING MODEL VERSION TABLE IN HUB...")
    input("PRESS ENTER IF YOU ARE SURE YOU WANT TO CONTINUE:")

    entity, project, _ = Model.version_table_path.split("/")
    with wandb.init(entity=entity, project=project) as run:  # type: ignore
        table_art = wandb.Artifact("model-tables", type="tables")
        with Path("versions.csv").open("w") as v:
            v.write("")
        table_art.add_file("versions.csv")
        run.log_artifact(table_art)
        run.link_artifact(table_art, Model.version_table_path)
    logging.warn("MODEL VERSION TABLE DELETED!")


def main(args: argparse.Namespace) -> None:
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise ValueError(f"config file {str(config_path.resolve())} does not exist!")
        with config_path.open() as config_file:
            config = json.load(config_file)
    else:
        config = vars(args)

    configure_logging(logging.getLogger(), None, "info", True)

    api = wandb.Api()
    filters = (
        None
        if config["runs"] is None
        else {"$or": [{"config.experiment_name": run_name} for run_name in config["runs"]]}
    )
    entity = config["entity"]
    project = config["project"]
    logging.info("Syncing runs from:", f"{entity}/{project}")
    runs = api.runs(
        path=f"{entity}/{project}",
        filters=filters,
    )

    compare_metrics = parse_metrics(config["compare_metrics"])

    version_table = download_version_table(Model.version_table_path, api, no_exception=True)
    for run in tqdm(runs):
        try:
            if len(run.logged_artifacts()) == 0:
                logging.info(f"run {run.name} has no logged artifacts, skipping...")
                continue
            model_kwargs = extract_model_parameters(run)
            model_name = model_kwargs["model_name"]
            version_table = update_table(version_table, model_kwargs, run, compare_metrics, api)
        except Exception as e:
            logging.info(f"run {run.name} cannot be synced with hub due to error: {e}. skipping...")

    write_table(version_table, model_name)
    logging.info(f"Successfully synced {len(runs)} runs!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sync script of wandb runs and the model zoo")
    parser.add_argument("--config", "-c", default=None, type=str, help="path to config json file")
    parser.add_argument(
        "--runs", "-r", nargs="*", default=None, help="the list of runs to sync. If omitted, all runs are synced."
    )
    parser.add_argument("--entity", "-e", default=None, type=str)
    parser.add_argument("--project", "-p", default=None, type=str)
    parser.add_argument(
        "--delete-version-table",
        default=False,
        action="store_true",
        help="deletes the remote version table on the model hub. Use with caution!",
    )
    args = parser.parse_args()

    if args.delete_version_table:
        delete_model_version_table_in_hub()
        sys.exit(0)
    main(args)
