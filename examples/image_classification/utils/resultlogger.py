import logging
import csv
import torch
from torch.nn import Module
from pathlib import Path
from typing import Union

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


class ResultLogger():
    """Logs the training results in a csv file and via tensorboard."""

    def __init__(self, result_file: Union[str, None], tensorboard: bool, tensorboard_output: str) -> None:
        """creates result file and tensorboard output, activates result logging

        Args:
            result_file (Union[str, None]): path to result file. if omitted, no results will be logged.
            tensorboard (bool): toggles tensorboard logging.
            tensorboard_output (str): path to tensorboard output dir
        """
        self._set_result_file(result_file)
        self._set_tensorboard(tensorboard, tensorboard_output)

    def _set_result_file(self, result_file: Union[str, None]) -> None:
        if result_file:
            self._result_file = Path(result_file)
            self._result_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._result_file = None  # type: ignore
            logging.warning("No results file set, training results will not be stored!")

    def _set_tensorboard(self, tensorboard_activated: bool, output: str) -> None:
        self._tensorboard = None
        self._tensorboard_output = None

        if SummaryWriter is None:
            logging.warning("Can not enable tensorboard logging because the package 'tensorboardX' is not installed!")
            return

        if not tensorboard_activated:
            logging.warning("No tensorboard enabled!")
            return

        self._tensorboard_output = Path(output)
        if self._tensorboard_output.exists():
            logging.warning(
                "tensorboard output folder already exists! this might cause unexpected result logging behaviour!")
        self._tensorboard_output.parent.mkdir(parents=True, exist_ok=True)
        self._tensorboard = SummaryWriter(self._tensorboard_output)

    def log_model(self, model: Module, example_input: torch.Tensor) -> None:
        """adds model graph to tensorboard.

        Args:
            model (Module): model to be visualized in tensorboard
            example_input (torch.Tensor): an example input (e.g. train data) to infer tensor dimensions throughout the
                model.
        """
        if not self._tensorboard:
            return

        self._tensorboard.add_graph(model, example_input)
        self._tensorboard.flush()

    def log_result(self, tensorboard: bool = False, log: bool = True, **kwargs: dict) -> None:
        """writes the values of the kwargs into a csv file while updating its header row.

        Args:
            tensorboard (bool, optional): toggles wheather the log values shall be outputed to tensorboard as well.
                Defaults to False.
            log (bool, optional): toggles wheather there should be a value log entry in default log files.
                Defaults to True.
        """
        if not self._result_file:
            return

        if log:
            logging.info(f"training results: {kwargs}")
        if tensorboard:
            self.tensorboard_results(**kwargs)  # type: ignore

        create_header = False
        if not self._result_file.exists():
            create_header = True

        with self._result_file.open("a+") as result_file:
            file_header = csv.DictReader(result_file).fieldnames
            if file_header:
                file_header = list(set().union(file_header, kwargs.keys()))  # type: ignore
            else:
                file_header = list(kwargs.keys())
                if create_header:
                    csv.writer(result_file).writerow(file_header)
            logging.debug(f"field names: {file_header}")
            dict_writer = csv.DictWriter(result_file, file_header, "0.0")
            dict_writer.writerow(kwargs)

    def tensorboard_results(
            self,
            step: Union[int, float],
            category: str = "",
            reverse_tag: bool = False,
            **kwargs: dict) -> None:
        """loggs the values in kwargs in tensorboard. category specifies the field to store the values in.
        Note: only scalar values supported right now.

        Args:
            category (str, optional): name of field to store the values in. Defaults to "".
            step (int or float, optional): step variable for scalar logging
            reverse_tag (bool, optional): reverses the tag for tensorboard logging (i.e. variable name / category
                vs. category / variable name)
        """
        if not self._tensorboard:
            return

        for key, value in kwargs.items():
            if isinstance(value, float) or isinstance(value, int):
                if reverse_tag:
                    tag = category + "/" + str(key)
                else:
                    tag = str(key) + "/" + category
                self._tensorboard.add_scalar(tag, value, step)
        self._tensorboard.flush()
