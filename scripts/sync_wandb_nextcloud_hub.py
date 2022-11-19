import pandas
import os
from pathlib import Path
import json
import wandb
import logging
import argparse

from bitorch.models import model_from_name

from examples.image_classification.datasets import dataset_from_name
from examples.image_classification.utils.arg_parser import create_argparser


def extract_model_parameters(run):
    print("config:", run.config)
    if "model_config" in run.config.keys():
        model_kwargs = run.config["model_config"]
    else:
        logging.info(
            "DEPRECATED: No model_config entry in run config found! Trying to reconstruct model parameters by parsing intial run command...")

        logging.debug("downloading run metadata...")
        run.file("wandb-metadata.json").download("/tmp", replace=True)

        metadata_path = Path("/tmp/wandb-metadata.json")
        if not metadata_path.exists():
            logging.error("metadata file could not be downloaded! skipping run...")
            return

        with metadata_path.open() as metadata_file:
            metadata = json.parse(metadata_file)

        parser, model_parser = create_argparser(metadata["args"])

        args_, unparsed_model_args = parser.parse_known_args()
        model_args_ = model_parser.parse_args(unparsed_model_args)

        dataset = dataset_from_name(args_.dataset)

        model_kwargs = vars(model_args_)
        model_kwargs["input_shape"] = dataset.shape
        model_kwargs["num_classes"] = dataset.num_classes
        model_kwargs["model_name"] = args_.model
        logging.debug(f"extracted model config: {model_kwargs}")
    return model_kwargs


def download_version_table(model_name):
    model_table_url = model_from_name(model_name).hub_version_table_url
    model_table_url = "https://nextcloud.hpi.de/s/bAAMM9PwTBe95Qe/download/example_table.csv"
    try:
        os.system(f"curl -T /tmp/version_table.csv {model_table_url}")
        version_table = pandas.read_csv("/tmp/version_table.csv")
    except:
        logging.info(f"could not retrieve model version table for {model_name}!")
        return pandas.DataFrame()

    return version_table


def update_table(version_table, model_kwargs, run, compare_metrics):
    ...


def write_table(version_table, model_name, table_base_url):
    ...


def main(args):
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise ValueError(f"config file {str(config_path.resolve())} does not exist!")
        with config_path.open() as config_file:
            config = json.parse(config_file)
    else:
        config = None

    print("args:", args, "wandb_entity:", args.wandb_entity)
    api = wandb.Api()
    filters = None if args.runs is None else {"$or": [{"config.experiment_name": run_name} for run_name in args.runs]}
    entity = config["wandb_entity"] if config is not None else args.wandb_entity
    project = config["wandb_project"] if config is not None else args.wandb_project
    print("path:", f"{entity}/{project}")
    runs = api.runs(
        path=f"{entity}/{project}",
        filters=filters,
    )

    metrics = config["metrics"] if config is not None else args.metrics
    compare_metrics = {}
    for metric in metrics:
        metric_name, mode = metric.split("/")
        if metric_name is None or mode is None:
            logging.error(
                f"metric cannot be parsed: {metric}. Needs to be in format <metric name>/<mode: min or max>! e.g.: accuracy/max")
            continue
        compare_metrics[metric_name] = mode

    table_base_url = config["tables_url"] if config is not None else args.tables_url

    for run in runs:

        model_kwargs = extract_model_parameters(run)

        model_name = model_kwargs["model_name"]
        version_table = download_version_table(model_name)
        version_table = update_table(version_table, model_kwargs, run, compare_metrics)

        write_table(version_table, model_name, table_base_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sync script of wandb runs and the model zoo")
    parser.add_argument("--config", "-c", default=None, type=str, help="path to config json file")
    parser.add_argument("--runs", "-r", nargs="*", default=None,
                        help="the list of runs to sync. If omitted, all runs are synced.")
    parser.add_argument("--wandb_entity", "-e", default=None, type=str)
    parser.add_argument("--wandb_project", "-p", default=None, type=str)
    args = parser.parse_args()
    main(args)
