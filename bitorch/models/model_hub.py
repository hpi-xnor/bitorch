from typing import Dict, Any
import wandb
import numbers
import pandas
import logging
import warnings
import torch


def convert_dtypes(data: dict) -> dict:
    """converts types of the values of dict so that they can be easily compared accross
    dataframes and csvs. converts all values that are not numerical to string.

    Args:
        data (dict): dict with values to be converted

    Returns:
        dict: dict with converted values
    """
    for key, value in data.items():
        if isinstance(value, list):
            value = tuple(value)
        if not isinstance(value, numbers.Number) and not isinstance(value, bool):
            data[key] = str(value)
    return data


def get_matching_row(version_table: pandas.DataFrame, model_kwargs: dict) -> pandas.DataFrame:
    """searches the version table dataframe for a row that matches model kwargs

    Args:
        version_table (pandas.DataFrame): the dataframe to search in
        model_kwargs (dict): the dict to search for. does not have to have key-value-pairs of each
        column of version_table, i.e. can be subset

    Returns:
        pandas.DataFrame: row with values in model_kwargs.keys() columns that are equal to model_kwargs values.
        if not existent, returns an empty dataframe.
    """
    model_kwargs = convert_dtypes(model_kwargs)
    with warnings.catch_warnings():
        model_kwargs_series = pandas.Series(model_kwargs)
        existing_row = version_table[(version_table[model_kwargs.keys()] == model_kwargs_series).all(1)]
    if existing_row.empty:
        return None
    return existing_row


def get_model_path(version_table: pandas.DataFrame, model_kwargs: dict, model_hub_base_path: str) -> str:
    """finds the matching row for model_kwargs in version table and path to model artifact for given configuration

    Args:
        version_table (pandas.DataFrame): version table with model configurations and corresponding model hub versions
        model_kwargs (dict): model configuration to search for
        model_hub_base_path (str): base path that contains model hub version of models

    Raises:
        RuntimeError: thrown if no matching model can be found in version table

    Returns:
        str: path to matching model hub artifact
    """
    matching_row = get_matching_row(version_table, model_kwargs)
    if matching_row is None:
        raise RuntimeError(
            f"No matching model found in hub with configuration: {model_kwargs}! You can train"
            " it yourself or try to load it from a local checkpoint!"
        )
    model_version = matching_row["model_hub_version"][0]
    return f"{model_hub_base_path}/{model_kwargs['model_name']}:v{model_version}"


def load_from_hub(
    model_version_table_path: str, model_hub_base_path: str, download_path: str = "/tmp", **model_kwargs: str
) -> torch.Tensor:
    """loads the model that matches the requested model configuration in model_kwargs from the model hub.

    Args:
        model_version_table_path (str): path to model version table on model hub
        model_hub_base_path (str): base path for model hub for downloading stored models
        download_path (str, optional): path to store the downloaded files. Defaults to "/tmp".

    Returns:
        torch.Tensor: state dict of downloaded model file
    """
    api = wandb.Api()
    version_table = download_version_table(model_version_table_path, api)
    model_path = get_model_path(version_table, model_kwargs, model_hub_base_path)
    logging.info("downloading model...")
    downloaded_model = api.artifact(model_path).get_path("model.ckpt").download(root=download_path)
    artifact = torch.load(downloaded_model)

    # true if artifact is a checkpoint from pytorch lightning
    if isinstance(artifact, dict):
        return lightning_checkpoint_to_state_dict(artifact)  # type: ignore
    return artifact


def lightning_checkpoint_to_state_dict(artifact: Dict[Any, Any]) -> Dict[Any, Any]:
    """converts a pytorch lightning checkpoint to a normal torch state dict

    Args:
        artifact (Dict[Any, Any]): dict containing a ['state_dict'] attribute

    Returns:
        Dict[Any, Any]: state dict for model
    """
    state_dict = artifact["state_dict"]
    # turns model._model.arg keys in state dict into _model.arg
    extracted_state_dict = {key[6:]: value for key, value in state_dict.items()}
    return extracted_state_dict


def download_version_table(model_table_path: str, api: wandb.Api, no_exception: bool = False) -> pandas.DataFrame:
    """downloads the newest version table from model hub.

    Args:
        model_table_path (str): path on hub to model version table
        api (wandb.Api): api to make download request with
        no_exception (bool, optional): weather exception shall be thrown if received version table is empty. Defaults to False.

    Raises:
        Exception: thrown if received version table is empty / cannot be downloaded and no_exception is False

    Returns:
        pandas.DataFrame: model version table
    """
    logging.info("downloading model version table from hub...")
    try:
        model_table = api.artifact(f"{model_table_path}:latest").get_path("versions.csv").download(root="/tmp")
        version_table = pandas.read_csv(model_table)
    except Exception as e:
        logging.info(f"could not retrieve model version table from {model_table_path}: {e}")
        if no_exception:
            logging.info("creating empty table...")
            return pandas.DataFrame()
        raise Exception(e)
    return version_table
