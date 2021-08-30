import logging
import csv
from pathlib import Path
from typing import Union
from tensorboardX import SummaryWriter


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
        if tensorboard_activated:
            self._tensorboard_output = Path(output)
            if self._tensorboard_output.exists():
                logging.warning(
                    "tensorboard output folder already exists! this might cause unexpected result logging behaviour!")
            self._tensorboard_output.parent.mkdir(parents=True, exist_ok=True)
            self._tensorboard = SummaryWriter(self._tensorboard_output)
        else:
            logging.warning("No tensorboard enabled!")

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

    def tensorboard_results(self, category: str = "", **kwargs: dict) -> None:
        """loggs the values in kwargs in tensorboard. category specifies the field to store the values in.
        Note: only scalar values supported right now.

        The first value in kwargs will be used as 'step' value, i.e. the x-value in scalar charts (e.g. epoch or
        batch num)

        Args:
            category (str, optional): name of field to store the values in. Defaults to "".
        """
        if not self._tensorboard:
            return

        step_variable = list(kwargs.values())[0]
        for key, value in list(kwargs.items())[1:]:
            if isinstance(value, float) or isinstance(value, int):
                self._tensorboard.add_scalar(category + "/" + str(key), value, step_variable)
        self._tensorboard.flush()
