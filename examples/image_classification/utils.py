import logging
import csv
import shutil
import os
import coloredlogs
import torch
import sys
import time
import subprocess
from math import floor
from pathlib import Path
from tensorboardX import SummaryWriter


def set_logging(log_file, log_level, output_stdout):

    log_level_name = log_level.upper()
    log_level = getattr(logging, log_level_name)
    logging.setLevel(log_level)

    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s [%(filename)s > %(funcName)s() > %(lineno)s]: %(message)s')
    coloredlogs.install()

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging_format)
        logging.addHandler(file_handler)

    output_stdout = output_stdout
    if output_stdout:
        stream = logging.StreamHandler()
        stream.setLevel(log_level)
        stream.setFormatter(logging_format)
        logging.addHandler(stream)


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

    def log_result(self, tensorboard=True, **kwargs):
        if not self._result_file:
            return

        logging.info("training results: {kwargs}")
        if tensorboard:
            self.tensorboard_results(kwargs)

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
            checkpoint_path = self._store_dir / f"checkpoint_epoch_{epoch}.pth"
        else:
            checkpoint_path = self._store_dir / f"{checkpoint_name}.pth"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }, checkpoint_path)

        if self._keep_count > 0:
            existing_checkpoints = list(self._store_dir.iterdir())
            sorted_checkpoints = sorted(existing_checkpoints, key=lambda checkpoint: checkpoint.stat().st_ctime)
            checkpoints_to_delete = sorted_checkpoints[self._keep_count:]

            for checkpoint in checkpoints_to_delete:
                checkpoint.unlink()

    def load_checkpoint(self, path, model, optimizer, lr_scheduler, fresh_start=False):

        if not path or not Path(path).exists():
            raise ValueError("checkpoint loading path not given or not existing!")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
        if fresh_start:
            epoch = 0
        else:
            epoch = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        return model, optimizer, lr_scheduler, epoch


class ExperimentCreator():

    project_root = "../.."

    def __init__(self, experiment_name, experiment_dir):
        self._experiment_name = experiment_name
        if not self._experiment_name:
            self._aquire_name()
        self._experiment_dir = Path(experiment_dir) / self._eperiment_name
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Experiment will be created in {self._experiment_dir}")

    def _extract_run_args(self, parser, args, model_parser, model_args):
        run_args = {}
        args_dict = vars(args)
        actions = parser._get_optional_actions()

        for action in actions:
            run_args[action.option_strings[0]] = args_dict[action.dest]

        model_args_dict = vars(model_args)
        actions = model_parser._get_optional_actions()

        for action in actions:
            run_args[action.option_strings[0]] = model_args_dict[action.dest]

        run_args["--log_file"] = self._experiment_dir / (f"{self._eperiment_name}.log")
        run_args["--result_file"] = self._experiment_dir / (f"{self._eperiment_name}.csv")
        run_args["--tensorboard_output"] = self._experiment_dir / "runs"
        run_args["--checkpoint_dir"] = self._experiment_dir / "checkpoints"

        if "--experiment" in run_args:
            del run_args["--experiment"]
        if "--experiment-dir" in run_args:
            del run_args["--experiment-dir"]
        if "--experiment-name" in run_args:
            del run_args["--experiment-name"]

        for key, value in run_args.items():
            if isinstance(value, bool):
                if not value:
                    del run_args[key]
                else:
                    run_args[key] = ""

        return run_args

    def _create_run_file(self, run_file_path, run_args, script_path):
        script_cli_args = [value for entry in run_args.items() for value in entry]
        script_execution_call = (
            f"python3 {str(script_path)}"
            " \\\n\t"
            (" \\\n\t".join(script_cli_args))
        )
        logging.info(f"script will be started with the following call: {script_execution_call}")

        with run_file_path.open("w") as run_file:
            run_file.write("#! /bin/bash\n")
            run_file.write("cd \"${0%/*}\"\n")
            run_file.write(script_execution_call)

        os.system(f"chmod 777 {str(run_file_path.resolve())}")

    def create(self, parser, args, model_parser, model_args):
        run_args = self._extract_run_args(parser, args, model_parser, model_args)

        shutil.copy(Path(self.project_root) / "*", self._experiment_dir / "code")

        script_path = Path(__file__).resolve()
        root_path = Path(self.project_root)
        relative_script_path = Path(os.path.relpath(script_path, start=root_path))

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
            f"progress: {self._fraction_to_percent(self._current_iteration / self._num_iterations)}, "
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
        self._current_iteration = 0
        self._start_time = time.time()

    def __exit__(self):
        self._current_iteration += 1
        time_diff = time.time() - self._start_time
        self._abs_time += time_diff
        if (self._current_iteration % self._log_interval) == 0:
            self._log_eta()
