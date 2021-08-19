import logging
import csv
import os
import torch
import sys
import shutil
import time
import subprocess
from math import floor
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR, _LRScheduler
from typing import Union, Optional
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from pathlib import Path
from tensorboardX import SummaryWriter

sys.path.append("../..")

from bitorch.optimization.radam import RAdam  # noqa: E402


def set_logging(log_file, log_level, output_stdout):
    logger = logging.getLogger()

    log_level_name = log_level.upper()
    log_level = getattr(logging, log_level_name)
    logger.setLevel(log_level)

    # coloredlogs.install(logger)
    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s [%(filename)s : %(funcName)s() : l. %(lineno)s]: %(message)s')

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging_format)
        logger.addHandler(file_handler)

    if output_stdout:
        stream = logging.StreamHandler()
        stream.setLevel(log_level)
        stream.setFormatter(logging_format)
        logger.addHandler(stream)


class ResultLogger():
    def __init__(self, result_file, tensorboard, tensorboard_output):
        self._set_result_file(result_file)
        self._set_tensorboard(tensorboard, tensorboard_output)

    def _set_result_file(self, result_file):
        self._result_file = result_file
        if self._result_file:
            self._result_file = Path(self._result_file)
            self._result_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            logging.warning("No results file set, training results will not be stored!")

    def _set_tensorboard(self, tensorboard_activated, output):
        self._tensorboard = None
        self._tensorboard_output = None
        if tensorboard_activated:
            self._tensorboard_output = Path(output)
            self._tensorboard_output.parent.mkdir(parents=True, exist_ok=True)
            self._tensorboard = SummaryWriter(self._tensorboard_output)
        else:
            logging.warning("No tensorboard enabled!")

    def log_result(self, tensorboard=False, log=True, **kwargs):
        if not self._result_file:
            return

        if log:
            logging.info(f"training results: {kwargs}")
        if tensorboard:
            self.tensorboard_results(kwargs)

        with self._result_file.open("a+") as result_file:
            file_header = csv.DictReader(result_file).fieldnames
            if file_header:
                file_header = list(set().union(file_header, kwargs.keys()))
            else:
                file_header = kwargs.keys()
            logging.debug(f"field names: {file_header}")
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


class CheckpointManager():
    def __init__(self, checkpoint_store_dir, keep_count):
        self._store_dir = checkpoint_store_dir
        if self._store_dir:
            self._store_dir = Path(self._store_dir)
            self._store_dir.mkdir(parents=True, exist_ok=True)
        else:
            logging.warn("No checkpoint store dir given, checkpoint storing disabled!")
        self._keep_count = keep_count

    def store_model_checkpoint(self, model, optimizer, lr_scheduler, epoch, checkpoint_name=None):

        if not checkpoint_name:
            checkpoint_path = self._store_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        else:
            checkpoint_path = self._store_dir / f"{checkpoint_name}.pth"
        logging.debug(f"storing checkpoint {checkpoint_path}...")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None
        }, checkpoint_path)

        if self._keep_count > 0:
            existing_checkpoints = list(self._store_dir.iterdir())
            sorted_checkpoints = sorted(existing_checkpoints, key=lambda checkpoint: checkpoint.stat().st_ctime)
            logging.debug(f"got sorted checkpoints: {sorted_checkpoints}")
            checkpoints_to_delete = sorted_checkpoints[:-self._keep_count]
            logging.debug(f"got oldest checkpoints: {checkpoints_to_delete}")

            for checkpoint in checkpoints_to_delete:
                checkpoint.unlink()

    def load_checkpoint(self, path, model, optimizer, lr_scheduler, fresh_start=False):

        if not path or not Path(path).exists():
            raise ValueError("checkpoint loading path not given or not existing!")
        logging.debug(f"loading checkpoint {path}....")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
        if fresh_start:
            epoch = 0
            logging.info("making a fresh start with pretrained model....")
        else:
            epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            if lr_scheduler and checkpoint["lr_scheduler"]:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        return model, optimizer, lr_scheduler, epoch


class ExperimentCreator():

    project_root = "../.."

    project_code = [
        "bitorch",
        "examples",
        "tests",
        "setup.cfg",
        "mypy.ini",
        "tests",
        "requirements-dev.txt",
        "requirements.txt",
    ]

    def __init__(self, experiment_name, experiment_dir, main_script_path):
        self._experiment_name = experiment_name
        if not self._experiment_name:
            self._acquire_name()
        # Todo: throw error if experiment dir is subdir of project code files / directories (causes loop in code copy)
        self._experiment_dir = Path(experiment_dir) / self._experiment_name
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Experiment will be created in {self._experiment_dir}")
        self._main_script_path = main_script_path

    def _acquire_name(self):
        while not self._experiment_name:
            print(
                "No experiment name given! Please enter a meaningful experiment name (e.g. new-resnet-architecture, "
                "sign-clipping-2.0, etc.). This will be the name of the experiment directory as well as log files, "
                "result file, etc."
            )
            self._experiment_name = input("experiment name > ")
            print("got experiment name: >", self._experiment_name, "<")

    def _extract_run_args(self, parser, args, model_parser, model_args):
        run_args = {}
        args_dict = vars(args)
        actions = parser._get_optional_actions()

        for action in actions:
            if "--help" not in action.option_strings:
                run_args[action.option_strings[0]] = args_dict[action.dest]

        model_args_dict = vars(model_args)
        actions = model_parser._get_optional_actions()

        for action in actions:
            if "help" not in action.option_strings:
                run_args[action.option_strings[0]] = model_args_dict[action.dest]

        run_args["--log-file"] = self._experiment_dir / (f"{self._experiment_name}.log")
        run_args["--result-file"] = self._experiment_dir / (f"{self._experiment_name}.csv")
        run_args["--tensorboard-output"] = self._experiment_dir / "runs"
        run_args["--checkpoint-dir"] = self._experiment_dir / "checkpoints"

        if "--experiment" in run_args:
            del run_args["--experiment"]
        if "--experiment-dir" in run_args:
            del run_args["--experiment-dir"]
        if "--experiment-name" in run_args:
            del run_args["--experiment-name"]
        if "--checkpoint-load" in run_args:
            del run_args["--checkpoint-load"]

        keys_to_delete = []
        for key, value in run_args.items():
            if isinstance(value, bool):
                if not value:
                    keys_to_delete.append(key)
                else:
                    run_args[key] = ""
            elif value is None:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del run_args[key]
        logging.debug(f"got run args: {run_args}")

        return run_args

    def _create_run_file(self, run_file_path, run_args, script_path):
        logging.debug("creating run file...")
        script_cli_args = [f"{key} {str(value)}" for key, value in run_args.items()]
        script_execution_call = (
            f"python3 {str(script_path.name)}" +
            " \\\n\t" +
            (" \\\n\t".join(script_cli_args))
        )
        logging.info(f"script will be started with the following call: {script_execution_call}")

        with run_file_path.open("w") as run_file:
            run_file.write("#! /bin/bash\n")
            run_file.write("cd \"${0%/*}\"\n")
            run_file.write("cd " + str(script_path.parent) + "\n")
            run_file.write(script_execution_call)

        os.system(f"chmod 777 {str(run_file_path.resolve())}")

    def create(self, parser, args, model_parser, model_args):
        run_args = self._extract_run_args(parser, args, model_parser, model_args)

        # shutil.copy(Path(self.project_root) / "*", self._experiment_dir / "code")
        code_path = (self._experiment_dir / "code/").resolve()
        tmp_path = Path("/tmp/code")
        root_path = Path(self.project_root)

        tmp_path.mkdir(exist_ok=True)
        logging.debug(f"now copying files to {tmp_path}....")
        for file_name in self.project_code:
            logging.debug(f"copying {file_name}...")
            file_path = (root_path / Path(file_name)).resolve()
            if file_path.is_dir():
                shutil.copytree(str(file_path), str(tmp_path / file_name), dirs_exist_ok=True)
            else:
                shutil.copy(str(file_path), str(tmp_path / file_name))
        logging.debug(f"copying files to {code_path}....")
        shutil.copytree(tmp_path, code_path, dirs_exist_ok=True)
        logging.debug(f"deleting {tmp_path}....")
        shutil.rmtree(tmp_path)

        script_path = Path(self._main_script_path).resolve()
        relative_script_path = Path("code") / Path(os.path.relpath(script_path, start=root_path))

        self._run_file_path = self._experiment_dir / "run.sh"
        self._create_run_file(self._run_file_path, run_args, relative_script_path)

    def run_experiment(self):
        logging.info(f"executing run file {str(self._run_file_path)}, exiting this script afterwards...")

        subprocess.Popen(self._run_file_path)
        sys.exit(0)


class ETAEstimator():
    def __init__(self, eta_file, log_interval, iterations=0):
        self._eta_file = eta_file
        if self._eta_file:
            self._eta_file = Path(self._eta_file)
            self._eta_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            logging.warn("no eta file given, eta will only be outputed via logging")
        self._log_interval = log_interval
        self._num_iterations = iterations
        self._current_iteration = 0
        self._start_time = 0
        self._abs_time = 0

    def set_iterations(self, num_iterations):
        self._num_iterations = num_iterations

    def _fraction_to_percent(self, fraction, decimals=3):
        return round(fraction * 100, decimals)

    def _seconds_to_timestamp(self, seconds):
        hours = floor(seconds / 3600)
        minutes = floor((seconds % 3600) / 60)
        seconds = round(seconds % 60, 1)
        return f"{str(hours)}h {str(minutes)}m {str(seconds)}s"

    def _log_eta(self):
        avg_time_per_unit = self._abs_time / self._current_iteration
        total_estimated_time = avg_time_per_unit * self._num_iterations
        remaining_estimated_time = total_estimated_time - self._abs_time

        log_msg = (
            f"average time per unit: {avg_time_per_unit} seconds, "
            f"progress: {self._fraction_to_percent(self._current_iteration / self._num_iterations)}%, "
            f"remaining estimated time: {self._seconds_to_timestamp(remaining_estimated_time)}"
        )

        logging.info(log_msg)
        if self._eta_file:
            try:
                with self._eta_file.open("a") as eta_file:
                    eta_file.write(log_msg + "\n")
            except IOError as e:
                logging.error(f"cannot write to eta file: {e}")
                raise IOError from e

    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, *args):
        self._current_iteration += 1
        time_diff = time.time() - self._start_time
        self._abs_time += time_diff
        if (self._current_iteration % self._log_interval) == 0:
            self._log_eta()


def create_optimizer(name: str, model: Module, lr: float, momentum: float) -> Optimizer:
    """creates the specified optimizer with the given parameters

    Args:
        name (str): str name of optimizer
        model (Module): the model used for training
        lr (float): learning rate
        momentum (float): momentum (only for sgd optimizer)

    Raises:
        ValueError: thrown if optimizer name not known

    Returns:
        Optimizer: the model optimizer
    """
    if name == "adam":
        return Adam(params=model.parameters(), lr=lr)
    elif name == "sgd":
        return SGD(params=model.parameters(), lr=lr, momentum=momentum)
    elif name == "radam":
        return RAdam(params=model.parameters(), lr=lr, degenerated_to_sgd=False)
    else:
        raise ValueError(f"No optimizer with name {name} found!")


def create_scheduler(
        scheduler_name: Optional[str],
        optimizer: Optimizer,
        lr_factor: float,
        lr_steps: Optional[list],
        epochs: int) -> Union[_LRScheduler, None]:
    """creates a learning rate scheduler with the given parameters

    Args:
        scheduler_name (Optional[str]): str name of scheduler or None, in which case None will be returned
        optimizer (Optimizer): the learning optimizer
        lr_factor (float): the learning rate factor
        lr_steps (Optional[list]): learning rate steps for the scheduler to take (only supported for step scheduler)
        epochs (int): number of scheduler epochs (only supported for cosine scheduler)

    Raises:
        ValueError: thrown if step scheduler was chosen but no steps were passed
        ValueError: thrown if scheduler name not known and not None

    Returns:
        Union[_LRScheduler, None]: either the learning rate scheduler object or None if scheduler_name was None
    """
    if scheduler_name == "step":
        if not lr_steps:
            raise ValueError("step scheduler chosen but no lr steps passed!")
        return MultiStepLR(optimizer, lr_steps, lr_factor)
    elif scheduler_name == "exponential":
        return ExponentialLR(optimizer, lr_factor)
    elif scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, epochs)
    elif not scheduler_name:
        return None
    else:
        raise ValueError(f"no scheduler with name {scheduler_name} found!")
