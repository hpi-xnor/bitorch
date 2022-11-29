import pandas
import os
from pathlib import Path
import json
import wandb
import logging
import argparse
import datetime

from bitorch.models import model_from_name

from examples.image_classification.datasets import dataset_from_name
from examples.image_classification.utils.arg_parser import create_argparser

INFINITY = 1e4

def add_missing_columns(*key_lists, init_values=None, table):
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
            metadata = json.load(metadata_file)
        
        print

        parser, model_parser = create_argparser(metadata["args"])

        args_, unparsed_model_args = parser.parse_known_args(metadata["args"])
        model_args_ = model_parser.parse_args(unparsed_model_args)

        dataset = dataset_from_name(args_.dataset)

        model_kwargs = vars(model_args_)
        model_kwargs["input_shape"] = dataset.shape
        model_kwargs["num_classes"] = dataset.num_classes
        model_kwargs["model_name"] = args_.model
        logging.debug(f"extracted model config: {model_kwargs}")
    return model_kwargs


def download_version_table(model_name, api):
    model_table_path = model_from_name(model_name).version_table_path
    try:
        # os.system(f"curl -T /tmp/version_table.csv {model_table_url}")
        model_table = api.artifact(model_table_path).get_path(f"{model_name}.csv").download(root="/tmp")
        version_table = pandas.read_csv(model_table)
    except:
        logging.info(f"could not retrieve model version table for {model_name}!")
        return pandas.DataFrame()

    return version_table

def upload_model_to_registry(run, api):
    run.link_artifact()


def update_table(version_table, model_kwargs, run, compare_metrics, api):
    for key in model_kwargs.keys():
        if key not in version_table.columns:
            version_table[key] = None
    
    version_table = add_missing_columns(
        model_kwargs.keys(), compare_metrics.keys(), ["time uploaded", "model_registry_version"],
        init_values=[None, [-INFINITY if mode == "min" else +INFINITY for mode in compare_metrics.values()], ["", "latest"]],
        table=version_table
    )
    existing_row = version_table[(version_table == pandas.Series(model_kwargs)).all(1)]
    if existing_row.empty:
        model_kwargs.update(dict(run.summary))
        upload_model_to_registry(run, api)
        version_table = pandas.concat([version_table, pandas.DataFrame([model_kwargs])])
    else:
        ...


def write_table(version_table, model_name, api):
    version_table.to_csv(f"/tmp/{model_name}.csv")
    model_table_path = model_from_name(model_name).version_table_path
    api.artifact(model_table_path).add_file(f"/tmp/{model_name}.csv").save()


def main(args):
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise ValueError(f"config file {str(config_path.resolve())} does not exist!")
        with config_path.open() as config_file:
            config = json.load(config_file)
    else:
        config = vars(args)

    print("args:", args, "entity:", config["entity"])
    api = wandb.Api()
    filters = None if config["runs"] is None else {"$or": [{"config.experiment_name": run_name} for run_name in config["runs"]]}
    entity = config["entity"]
    project = config["project"]
    print("path:", f"{entity}/{project}")
    runs = api.runs(
        path=f"{entity}/{project}",
        filters=filters,
    )

    metrics = config["compare_metrics"]
    compare_metrics = {}
    for metric in metrics:
        parts = metric.split("/")
        metric_name = "/".join(parts[:-1])
        mode = parts[-1]
        if metric_name is None or mode is None or mode not in ["min", "max"]:
            logging.error(
                f"metric cannot be parsed: {metric}. Needs to be in format <metric name>/<mode: min or max>! e.g.: accuracy/max")
            continue
        compare_metrics[metric_name] = mode
    
    
    for idx, run in enumerate(runs):
    # for run in runs:
        if idx < 15:
            continue

        model_kwargs = extract_model_parameters(run)

        model_name = model_kwargs["model_name"]
        version_table = download_version_table(model_name, api)
        version_table = update_table(version_table, model_kwargs, run, compare_metrics, api)

        write_table(version_table, model_name, api)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sync script of wandb runs and the model zoo")
    parser.add_argument("--config", "-c", default=None, type=str, help="path to config json file")
    parser.add_argument("--runs", "-r", nargs="*", default=None,
                        help="the list of runs to sync. If omitted, all runs are synced.")
    parser.add_argument("--entity", "-e", default=None, type=str)
    parser.add_argument("--project", "-p", default=None, type=str)
    args = parser.parse_args()
    main(args)
