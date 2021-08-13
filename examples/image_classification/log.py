import logging
import argparse
import csv
from pathlib import Path
from tensorboardX import SummaryWriter


class Logger():

    def __init__(self, args: argparse.Namespace):
        self._logger = logging.getLogger(__name__)

        log_level_name = args.log_level.upper()
        self._log_level = getattr(logging, log_level_name)
        self._logger.setLevel(self._log_level)

        self._logging_format = logging.Formatter(
            '%(asctime)s - %(levelname)s [%(filename)s > %(funcName)s() > %(lineno)s]: %(message)s')

        self._set_log_file(args.log_file)
        self._add_stdout(args.stdout)
        self._set_result_file(args.result_file)
        self._set_tensorboard(args.tensorboard, args.tensorboard_output)

    def _set_log_file(self, log_file):
        self._log_file = log_file
        if self._log_file:
            self._log_file = Path(self._log_file)
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self._log_file)
            file_handler.setLevel(self._log_level)
            file_handler.setFormatter(self._logging_format)
            self._logger.addHandler(file_handler)

    def _add_stdout(self, output_stdout):
        self._output_stdout = output_stdout
        if self._output_stdout:
            stream = logging.StreamHandler()
            stream.setLevel(self._log_level)
            stream.setFormatter(self._logging_format)
            self._logger.addHandler(stream)

    def _set_result_file(self, result_file):
        self._result_file = result_file
        if self._result_file:
            self._result_file = Path(self._result_file)
            self._result_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.warning("No results file set, training results will not be stored!")

    def _set_tensorboard(self, tensorboard_activated, output):
        self._tensorboard = None
        self._tensorboard_output = None
        if tensorboard_activated:
            self._tensorboard_output = Path(output)
            self._tensorboard_output.parent.mkdir(parents=True, exist_ok=True)
            self._tensorboard = SummaryWriter(self._tensorboard_output)
        else:
            self.warning("No tensorboard enabled!")

    def _args_to_string(self, args):
        return " ".join([str(argument) for argument in args])

    def _kwargs_to_string(self, kwargs):
        return " ".join("[" + str(key) + ": " + str(value) + "]" for key, value in kwargs.items())

    def debug(self, *args, **kwargs):
        logging.debug(self._args_to_string(args) + " " + self._kwargs_to_string(kwargs))

    def info(self, *args, **kwargs):
        logging.info(self._args_to_string(args) + " " + self._kwargs_to_string(kwargs))

    def warning(self, *args, **kwargs):
        logging.warning(self._args_to_string(args) + " " + self._kwargs_to_string(kwargs))

    def error(self, *args, **kwargs):
        logging.error(self._args_to_string(args) + " " + self._kwargs_to_string(kwargs))

    def critical(self, *args, **kwargs):
        logging.critical(self._args_to_string(args) + " " + self._kwargs_to_string(kwargs))

    def log_result(self, tensorboard=True, **kwargs):
        if not self._result_file:
            return

        self.info("training results:", **kwargs)

        with self._result_file.open("a+") as result_file:
            file_header = list(set().union(csv.reader(result_file), kwargs.keys()))
            dict_writer = csv.DictWriter(result_file, file_header, "0.0")
            dict_writer.writerow(kwargs)

    def tensorboard_results(self, category="", **kwargs):
        if not self._tensorboard:
            return

        step_variable = list(kwargs.values())[0]
        for key, value in list(kwargs.items())[1:]:
            if isinstance(value, float) or isinstance(value, int):
                self._tensorboard.add_scalar(category + "/" + str(key), value, step_variable)
        self._tensorboard.flush()

    def store_model_checkpoint(self, model, optimizer, lr_scheduler, epoch):
        pass