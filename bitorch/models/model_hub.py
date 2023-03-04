import copy
from pathlib import Path
from typing import Dict, Any, Union, Tuple
import numbers
import pandas
import logging
import warnings
import torch
import base64
import hashlib
from torch.nn import Module, Sequential, Identity
from torchvision.datasets.utils import download_url


def get_children(model: torch.nn.Module, name=[]):
    # get children form model!
    children = list(model.named_children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return [[name, model]]
    else:
        # look for children from children... to the last child!
        for child_name, child in children:
            flatt_children.extend(get_children(child, name=name + [child_name]))
    return flatt_children


def set_layer_in_model(model: Module, layer_names: list = [], layer: Module = None):
    """sets a layer in a model at the given layer_names.

    Args:
        model (Module): the model
        layer_names (list): the layer names
        layer (Module): the layer to set
    """
    if len(layer_names) == 0:
        return
    if len(layer_names) == 1:
        model._modules[layer_names[0]] = layer
        return
    set_layer_in_model(model._modules[layer_names[0]], layer_names[1:], layer)


def pop_first_layers(model: Module, num_layers: int = 1) -> Module:
    """pops the first num_layers layers from the model.

    Args:
        model (Module): the model
        num_layers (int): number of layers to pop

    Returns:
        Module: the model without the first num_layers layers
    """
    if num_layers == 0:
        return model
    layer_list = get_children(model)
    if len(layer_list) == 0:
        return model
    last_output_size = None
    last_name = ""
    last_layer = None
    i = 0
    for name, layer in layer_list:
        set_layer_in_model(model, name, Identity())
        if hasattr(layer, "out_features") or hasattr(layer, "out_channels"):
            i += 1
            last_name = name
            last_output_size = layer.out_features if hasattr(layer, "out_features") else layer.out_channels
            last_layer = copy.deepcopy(layer)
            if i == num_layers:
                break
    return last_name, last_output_size, last_layer


def pop_last_layers(model: Module, num_layers: int = 1) -> Module:
    """pops the last num_layers layers which transform the input/output size from the model.

    Args:
        model (Module): the model
        num_layers (int): number of layers to pop

    Returns:
        Module: the model without the last num_layers layers
    """
    if num_layers == 0:
        return model
    layer_list = get_children(model)
    layer_list.reverse()
    if len(layer_list) == 0:
        return model
    last_input_size = None
    last_layer_name = ""
    last_layer = None
    i = 0
    for name, layer in layer_list:
        set_layer_in_model(model, name, Identity())
        if hasattr(layer, "in_features") or hasattr(layer, "in_channels"):
            last_layer_name = name
            last_input_size = layer.in_features if hasattr(layer, "in_features") else layer.in_channels
            last_layer = copy.deepcopy(layer)
            i += 1
            if i == num_layers:
                break
    return last_layer_name, last_input_size, last_layer


def get_output_size(model: Module) -> int:
    """returns the output size of the model for the given input shape.

    Args:
        model (Module): the model

    Returns:
        int: the output size
    """
    for _, layer in reversed(get_children(model)):  # type: ignore
        if hasattr(layer, "out_features"):
            return layer.out_features, type(layer)
        if hasattr(layer, "out_channels"):
            return layer.out_channels, type(layer)
    return -1, None


def get_input_size(model: Module) -> int:
    """returns the input size of the model for the given input shape.

    Args:
        model (Module): the model

    Returns:
        int: the input size
    """
    for _, layer in get_children(model):  # type: ignore
        if hasattr(layer, "in_features"):
            return layer.in_features, type(layer)
        if hasattr(layer, "in_channels"):
            return layer.in_channels, type(layer)
    return -1, None


def _md5_hash_file(path: Path) -> Any:
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5


def _digest_file(path: Union[Path, str]) -> str:
    return base64.b64encode(_md5_hash_file(Path(path)).digest()).decode("ascii")


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


def get_model_path(version_table: pandas.DataFrame, model_kwargs: dict) -> Tuple[str, str]:
    """finds the matching row for model_kwargs in version table and path to model artifact for given configuration

    Args:
        version_table (pandas.DataFrame): version table with model configurations and corresponding model hub versions
        model_kwargs (dict): model configuration to search for

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
    model_url = matching_row["model_hub_url"].iloc[0]
    model_digest = matching_row["model_digest"].iloc[0]
    return model_url, model_digest


def load_from_hub(
    model_version_table_path: str, download_path: str = "bitorch_models", **model_kwargs: str
) -> torch.Tensor:
    """loads the model that matches the requested model configuration in model_kwargs from the model hub.

    Args:
        model_version_table_path (str): path to model version table on model hub
        download_path (str, optional): path to store the downloaded files. Defaults to "/tmp".

    Returns:
        torch.Tensor: state dict of downloaded model file
    """
    Path(download_path).mkdir(parents=True, exist_ok=True)

    version_table = download_version_table(model_version_table_path)
    model_path, model_digest = get_model_path(version_table, model_kwargs)
    model_checksum = model_path.split("/")[-1]
    model_local_path = Path(f"{download_path}/{model_checksum}")

    if not model_local_path.exists() or _digest_file(str(model_local_path)) != model_digest:
        logging.info("downloading model...")
        download_url(model_path, model_local_path.parent, model_local_path.name, model_checksum)
        logging.info("Model downloaded!")
    else:
        logging.info(f"Using already downloaded model at {model_local_path}")
    artifact = torch.load(model_local_path, map_location="cpu")

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
    state_dict = {key: value for key, value in artifact["state_dict"].items() if "model" in key}

    for key in state_dict.keys():
        assert key.startswith("model."), f"Unexpected malformed static dict key {key}."

    # turns model._model.arg keys in state dict into _model.arg
    extracted_state_dict = {key[6:]: value for key, value in state_dict.items()}
    return extracted_state_dict


def download_version_table(model_table_path: str, no_exception: bool = False) -> pandas.DataFrame:
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
        download_url(model_table_path, "/tmp", "bitorch_model_version_table.csv")
        version_table = pandas.read_csv("/tmp/bitorch_model_version_table.csv")
    except Exception as e:
        logging.info(f"could not retrieve model version table from {model_table_path}: {e}")
        if no_exception:
            logging.info("creating empty table...")
            return pandas.DataFrame()
        raise Exception(e)
    return version_table
