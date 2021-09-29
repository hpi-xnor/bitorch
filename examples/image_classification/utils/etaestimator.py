import time
import logging
from pathlib import Path
from math import floor


class ETAEstimator():
    """estimates the total runtime of a training run. A maximum number of single iterations needs to be set. after that,
    time consumption of single iterations can be measured by with-call of this object"""

    def __init__(self, eta_file: str, log_interval: int, iterations: int = 0) -> None:
        """creates eta file, inits some attributes

        Args:
            eta_file (str): path to eta file. will be created.
            log_interval (int): number of iterations to wait between eta outputs.
            iterations (Union[int, None], optional): max number of iterations. can be set later via set_iterations
                method. Defaults to None.
        """
        if eta_file:
            self._eta_file = Path(eta_file)
            self._eta_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._eta_file = None  # type: ignore
            logging.warn("no eta file given, eta will only be printed to logs")
        self._log_interval = log_interval
        self._num_iterations = iterations
        self._current_iteration = 0
        self._start_time = 0.0
        self._abs_time = 0.0
        self._epoch_duration = 0.0
        self._epoch_start = 0.0
        self._iterations_per_second = 0.0

    def epoch_start(self) -> None:
        """can be called at the start of an epoch. stores the current time as the start of an epoch"""
        self._epoch_start = time.time()

    def epoch_end(self) -> None:
        """can be called at the end of an epoch. stores the epoch duration as the time difference between stored epoch
        start and current time."""
        self._epoch_duration = time.time() - self._epoch_start
        self._epoch_start = 0.0

    def epoch_duration(self) -> float:
        """getter method for epoch duration.

        Returns:
            float: duration of an epoch in seconds.
        """
        return self._epoch_duration

    def set_iterations(self, num_iterations: int) -> None:
        """setter of iterations

        Args:
            num_iterations (int): number of maximum number of iterations.
        """
        self._num_iterations = num_iterations

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """converts a given number of seconds to a time stamp with hours minutes and seconds.

        Args:
            seconds (float): number of seconds

        Returns:
            str: time stamp string
        """
        hours = floor(seconds / 3600)
        minutes = floor((seconds % 3600) / 60)
        seconds = round(seconds % 60, 1)
        return f"{str(hours)}h {str(minutes)}m {str(seconds)}s"

    def eta(self) -> str:
        """creates an eta time string

        Returns:
            str: eta time string
        """
        avg_time_per_iteration = self._abs_time / self._current_iteration
        total_estimated_time = avg_time_per_iteration * self._num_iterations
        remaining_estimated_time = total_estimated_time - self._abs_time

        return self._seconds_to_timestamp(remaining_estimated_time)

    def iterations_per_second(self) -> float:
        """returns the estimated number of samples per second based on the last iteration

        Returns:
            float: estimated iterations per second
        """
        return self._iterations_per_second

    def summary(self) -> str:
        """creates an eta log message and outputs it both to logging and eta file.

        Raises:
            IOError: thrown if eta file cannot be written to.

        Returns:
            str: the log message (for e.g. logging)
        """
        avg_time_per_iteration = self._abs_time / self._current_iteration
        progress = round(self._current_iteration / self._num_iterations * 100, 3)

        log_msg = (
            f"average time per iteration: {avg_time_per_iteration} seconds, "
            f"progress: {progress}%, "
            f"remaining estimated time: {self.eta()}"
        )

        if self._eta_file:
            try:
                with self._eta_file.open("a") as eta_file:
                    eta_file.write(log_msg + "\n")
            except IOError as e:
                logging.error(f"cannot write to eta file: {e}")
                raise IOError from e
        return log_msg

    def __enter__(self) -> None:
        """executed when entering with statement.

        Raises:
            ValueError: thrown if max number of iterations / iterations not specified yet.
        """
        if self._num_iterations == 0:
            raise ValueError("number of iterations is not specified!")
        self._start_time = time.time()

    def __exit__(self, *args: list) -> None:
        """executed when leaving with statement. logs eta if log interval is over."""
        self._current_iteration += 1
        time_diff = time.time() - self._start_time
        self._iterations_per_second = 1.0 / time_diff
        self._abs_time += time_diff
        if (self._current_iteration % self._log_interval) == 0:
            self.summary()
