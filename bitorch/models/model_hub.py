import wandb
import numbers
import pandas
import logging
import warnings
import torch

def convert_dtypes(data):
    for key, value in data.items():
        if isinstance(value, list):
            value = tuple(value)
        if not isinstance(value, numbers.Number) and not isinstance(value, bool):
            data[key] = str(value)
    return data

def get_matching_row(version_table, model_kwargs):
    model_kwargs = convert_dtypes(model_kwargs)
    with warnings.catch_warnings():
        model_kwargs_series = pandas.Series(model_kwargs)
        existing_row = version_table[(version_table[model_kwargs.keys()] == model_kwargs_series).all(1)]
    if existing_row.empty:
        return None
    return existing_row

def get_model_path(version_table, model_kwargs, model_hub_base_path):
    matching_row = get_matching_row(version_table, model_kwargs)
    if matching_row is None:
        raise RuntimeError(f"No matching model found in hub with configuration: {model_kwargs}! You can train"
                           " it yourself or try to load it from a local checkpoint!")
    model_version = matching_row["model_hub_version"][0]
    return f"{model_hub_base_path}/{model_kwargs['model_name']}:v{model_version}"

def load_from_hub(model_version_table_path, model_hub_base_path, download_path="/tmp", **model_kwargs):
    api = wandb.Api()
    version_table = download_version_table(model_version_table_path, api)
    model_path = get_model_path(version_table, model_kwargs, model_hub_base_path)
    logging.info("downloading model...")
    downloaded_model = api.artifact(model_path).get_path("model.ckpt").download(root=download_path)
    artifact = torch.load(downloaded_model)
    
    # true if artifact is a checkpoint from pytorch lightning
    if isinstance(artifact, dict):
        return lightning_checkpoint_to_state_dict(artifact)
    return artifact

def lightning_checkpoint_to_state_dict(artifact):
    state_dict = artifact["state_dict"]
    # turns model._model.arg keys in state dict into _model.arg
    extracted_state_dict = {key[6:]: value for key, value in state_dict.items()}
    return extracted_state_dict

def download_version_table(model_table_path, api, no_exception=False):
    logging.info("downloading model version table from hub...")
    try:
        model_table = api.artifact(f"{model_table_path}:latest").get_path(f"versions.csv").download(root="/tmp")
        version_table = pandas.read_csv(model_table)
    except Exception as e:
        logging.info(f"could not retrieve model version table from {model_table_path}: {e}")
        if no_exception:
            logging.info("creating empty table...")
            return pandas.DataFrame()
        raise Exception(e)
    return version_table