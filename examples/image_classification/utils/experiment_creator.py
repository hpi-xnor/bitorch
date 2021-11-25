import logging
import shutil
import sys
import subprocess
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union


class ExperimentCreator():
    """Creates an experiment directory and runs scirpt in there. copys code, redirects logging, tensorboard and checkpoint
    output. Starts the script in that experiment directory and exits this programm."""

    # the root folder of this repository
    # (we need to navigate to the parent 4 times: utils, image_classification, examples, project root)
    project_root = Path(__file__).parent.parent.parent.parent
    assert (project_root / "bitorch").is_dir(), "The project root '{}' does not contain 'bitorch'.".format(project_root)

    # the files that will be copied for experiment execution.
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

    def __init__(self, experiment_name: Union[str, None], experiment_dir: str, main_script_path: str) -> None:
        """asks the user for an experiment name if none specified, creates experiment dir and makes sure its outside
        project root.

        Args:
            experiment_name (Union[str, None]): Name of the experiment run. if omitted, a dialog will be prompted.
            experiment_dir (str): path to where to create the experiment directory.
            main_script_path (str): path to main run script (for use in the experiment run.sh)

        Raises:
            ValueError: thrown if experiment dir is in a subdirectory of the directories that will be copied.
                (causes copy loop).
        """

        self._experiment_name = experiment_name
        if not self._experiment_name:
            self._acquire_name()
        self._experiment_dir = (Path(experiment_dir) / self._experiment_name).absolute()  # type: ignore

        # checks if experiment directory in one of the specified code files / directories
        for code_file in self.project_code:
            code_path = self.project_root / code_file
            if code_path in self._experiment_dir.parents:
                raise ValueError(f"experiment directory must not be in a subdirectory of {code_path.resolve()}!")

        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Experiment will be created in {self._experiment_dir}")
        self._main_script_path = main_script_path

    def _acquire_name(self) -> None:
        """User dialog to enter a meaningful experiment name"""
        while not self._experiment_name:
            print(
                "No experiment name given! Please enter a meaningful experiment name (e.g. new-resnet-architecture, "
                "sign-clipping-2.0, etc.). This will be the name of the experiment directory as well as log files, "
                "result file, etc."
            )
            self._experiment_name = input("experiment name > ")
            print("got experiment name: >", self._experiment_name, "<")

    def _extract_run_args(
            self,
            parser: ArgumentParser,
            args: Namespace,
            model_parser: ArgumentParser,
            model_args: Namespace) -> dict:
        """recreates the comand line arguments the main script was called with. also includes arguments that were not
        specified with their default values (for transparency reasons). redirects some options like log file, etc. to
        new experiment directory.

        Args:
            parser (ArgumentParser): parser of main arguments
            args (Namespace): main argument namespace
            model_parser (ArgumentParser): parser of model arguments
            model_args (Namespace): model argument namespace

        Returns:
            dict: the recreated and modified run args.
        """
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

        run_args["--log-file"] = self._experiment_dir / "experiment.log"
        run_args["--result-file"] = self._experiment_dir / "metrics.csv"
        run_args["--tensorboard-output"] = self._experiment_dir / "tblogs"
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

    def _create_run_file(self, run_file_path: Path, run_args: dict, script_path: Path) -> None:
        """creates a run.sh in experiment dir with script call.

        Args:
            run_file_path (Path): path to where run.sh shall be created
            run_args (dict): run arguments of main script
            script_path (Path): path to main script relative to experiment dir root.
        """
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

    def create(
            self,
            parser: ArgumentParser,
            args: Namespace,
            model_parser: ArgumentParser,
            model_args: Namespace) -> None:
        """creates the experiment. recreates the full script command line arguments, copies the code to expriment directory
        and creates a run.sh. logging (both standard and result), tensorboard output and checkpoint storing will be
        redirected to experiment directory.

        Args:
            parser (ArgumentParser): parser of main arguments
            args (Namespace): main argument namespace
            model_parser (ArgumentParser): parser of model arguments
            model_args (Namespace): model argument namespace
        """
        run_args = self._extract_run_args(parser, args, model_parser, model_args)

        code_path = (self._experiment_dir / "code/").resolve()

        logging.debug(f"now copying files to {code_path}....")
        for file_name in self.project_code:
            logging.debug(f"copying {file_name}...")
            file_path = (self.project_root / Path(file_name)).resolve()
            if file_path.is_dir():
                shutil.copytree(str(file_path), str(code_path / file_name), dirs_exist_ok=True)
            else:
                shutil.copy(str(file_path), str(code_path / file_name))

        script_path = Path(self._main_script_path).resolve()
        relative_script_path = Path("code") / Path(os.path.relpath(script_path, start=self.project_root))

        self._run_file_path = self._experiment_dir / "run.sh"
        self._create_run_file(self._run_file_path, run_args, relative_script_path)

    def run_experiment_in_subprocess(self) -> None:
        """runs the run.sh in experiment dir. exits this programm afterwards."""
        logging.info(f"executing run file {str(self._run_file_path)}, exiting this script afterwards...")

        subprocess.Popen(str(self._run_file_path))
