import warnings
import pandas
import os
from pathlib import Path
import json
import wandb
import logging
import argparse
import datetime
from typing import Any, Optional

from bitorch.models import model_from_name
from bitorch.models.base import Model

from examples.image_classification.datasets import dataset_from_name
# from examples.image_classification.utils.arg_parser import create_argparser
from importlib import import_module

warnings.filterwarnings("ignore")
INFINITY = 1e4

def configure_logging(logger: Any, log_file: Optional[str], log_level: str, output_stdout: bool) -> None:
    """configures logging module.

    Args:
        logger: the logger to be configured
        log_file (str): path to log file. if omitted, logging will be forced to stdout.
        log_level (str): string name of log level (e.g. 'debug')
        output_stdout (bool): toggles stdout output. will be activated automatically if no log file was given.
            otherwise if activated, logging will be outputed both to stdout and log file.
    """
    log_level_name = log_level.upper()
    log_level = getattr(logging, log_level_name)
    logger.setLevel(log_level)

    logging_format = logging.Formatter(
        "%(asctime)s - %(levelname)s [%(filename)s : %(funcName)s() : l. %(lineno)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file is not None:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging_format)
        logger.addHandler(file_handler)
    else:
        output_stdout = True

    if output_stdout:
        stream = logging.StreamHandler()
        stream.setFormatter(logging_format)
        logger.addHandler(stream)

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
    if "model_config" in run.config.keys():
        model_kwargs = run.config["model_config"]
    else:
        logging.info(
            "DEPRECATED: No model_config entry in run config found! Trying to reconstruct model parameters by parsing intial run command (experimental)...")

        logging.debug("downloading run metadata...")
        run.file("wandb-metadata.json").download("/tmp", replace=True)

        metadata_path = Path("/tmp/wandb-metadata.json")
        if not metadata_path.exists():
            logging.error("metadata file could not be downloaded! skipping run...")
            return

        with metadata_path.open() as metadata_file:
            metadata = json.load(metadata_file)
        
        if "dlrm" in metadata["program"]:
            create_argparser = import_module("examples.dlrm.utils.arg_parser").create_argparser
        else:
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

def extract_artifact_version(artifact, model_name):
    for alias in artifact._attrs["aliases"]:
        if alias["artifactCollectionName"] == model_name and alias["alias"][0] == "v" and alias["alias"][1:].isnumeric():
            return int(alias["alias"][1:])


def download_version_table(api):
    model_table_path = Model.version_table_path
    try:
        # os.system(f"curl -T /tmp/version_table.csv {model_table_url}")
        model_table = api.artifact(f"{model_table_path}:latest").get_path(f"versions.csv").download(root="/tmp")
        version_table = pandas.read_csv(model_table)
    except Exception as e:
        logging.info(f"could not retrieve model version table from {model_table_path}: {e}!")
        return pandas.DataFrame()

    return version_table

def upload_model_to_registry(run, model_name, api):
    model_registry_path = f"{model_from_name(model_name).model_registry_base_path}/{model_name}"
    run_artifact = run.logged_artifacts()[0]
    run_artifact.link(model_registry_path)
    uploaded_artifact = api.artifact(f"{model_registry_path}:latest")
    return extract_artifact_version(uploaded_artifact, model_name)

def compare(configA, configB, compare_metrics):
    for metric_name, mode in compare_metrics:
        if configA[metric_name] == configB[metric_name]:
            continue
        # this should be correct...
        return (configA[metric_name] < configB[metric_name]) == (mode == "max")

def add_model_to_version_table(model_kwargs, version_table, run, api):
        model_kwargs["model_registry_version"] = upload_model_to_registry(run, model_kwargs["model_name"], api)
        return pandas.concat([version_table, pandas.DataFrame([model_kwargs])])
    

def update_table(version_table, model_kwargs, run, compare_metrics, api):
    version_table = add_missing_columns(
        model_kwargs.keys(), dict(run.summary).keys(), [metric_name for metric_name, _ in compare_metrics], ["time uploaded", "model_registry_version"],
        init_values=[None, None, [-INFINITY if mode == "min" else +INFINITY for mode in [mode for _, mode in compare_metrics]], ["", "latest"]],
        table=version_table
    )
    
    print("comparing ", model_kwargs)
    # this removes a deprecation warning
    model_kwargs_series = pandas.Series(model_kwargs)
    model_kwargs.update(dict(run.summary))
    # model_kwargs_series, version_table = model_kwargs_series.align(version_table, axis=0, copy=False)
    
    existing_row = version_table[(version_table == model_kwargs_series).all(1)]
    model_kwargs["time uploaded"] = str(datetime.datetime.now())

    if existing_row.empty:
        logging.info("adding new model configuration to version table...")
        version_table = add_model_to_version_table(model_kwargs, version_table, run, api)
    else:
        existing_model_kwargs = existing_row.to_dict()
        existing_row_idx = (version_table == model_kwargs_series).all(1)
        
        # this prevents reuploading the same model
        compare_metrics.append(["time uploaded", "min"])
        new_model_better = compare(existing_model_kwargs, model_kwargs, compare_metrics)
        if new_model_better:
            logging.info("overwriting preexisting model configuration in version table with better version...")
            version_table = version_table.drop(existing_row_idx)
            version_table = add_model_to_version_table(model_kwargs, version_table, run, api)
        else:
            logging.info("better/older model with same config already exists in version table, skipping...")
    return version_table


def write_table(version_table, model_name, api):
    version_table.to_csv(f"/tmp/versions.csv")
    entity, project, _ = model_from_name(model_name).version_table_path.split("/")
    with wandb.init(entity=entity, project=project) as run:
        version_table_artifact = wandb.Artifact("model-tables", type="tables")
        version_table_artifact.add_file("/tmp/versions.csv")
        run.log_artifact(version_table_artifact)
        run.link_artifact(version_table_artifact, model_from_name(model_name).version_table_path)


def main(args):
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise ValueError(f"config file {str(config_path.resolve())} does not exist!")
        with config_path.open() as config_file:
            config = json.load(config_file)
    else:
        config = vars(args)
    
    configure_logging(logging.getLogger(), None, "info", True)

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
    compare_metrics = []
    for metric in metrics:
        parts = metric.split("/")
        metric_name = "/".join(parts[:-1])
        mode = parts[-1]
        if metric_name is None or mode is None or mode not in ["min", "max"]:
            logging.error(
                f"metric cannot be parsed: {metric}. Needs to be in format <metric name>/<mode: min or max>! e.g.: accuracy/max")
            continue
        compare_metrics.append([metric_name, mode])
    
    version_table = download_version_table(api)
    for idx, run in enumerate(runs):
    # for run in runs:
        if idx < 2:
            continue
        # try:
        if len(run.logged_artifacts()) == 0:
            logging.info(f"run {run.name} has no logged artifacts, skipping...")
            continue
        model_kwargs = extract_model_parameters(run)
        model_name = model_kwargs["model_name"]
        version_table = update_table(version_table, model_kwargs, run, compare_metrics, api)

    write_table(version_table, model_name, api)
        # except Exception as e:
        #     logging.error(f"could not sync run {run.name}:{e}. Skipping...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sync script of wandb runs and the model zoo")
    parser.add_argument("--config", "-c", default=None, type=str, help="path to config json file")
    parser.add_argument("--runs", "-r", nargs="*", default=None,
                        help="the list of runs to sync. If omitted, all runs are synced.")
    parser.add_argument("--entity", "-e", default=None, type=str)
    parser.add_argument("--project", "-p", default=None, type=str)
    args = parser.parse_args()
    main(args)
