"""Base classes and data containers."""

from __future__ import annotations

import logging

# Imports from standard library
from datetime import datetime
from logging import Logger, LogRecord
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import colorama

# Import from dependencies
import numpy as np
import tqdm
from colorama import Back, Fore, Style

# Import from package
if TYPE_CHECKING:
    from fm2prof.IniFile import IniFile


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler for tqdm package."""

    def __init__(self) -> None:
        """Instantiate a TqdmLoggingHandler."""
        super().__init__()

    def emit(self, record: LogRecord) -> None:
        """Write progressbar to logstream."""
        try:
            msg = self.format(record)
            if self.formatter.pbar:
                self.formatter.pbar.write(msg)
            else:
                stream = self.stream
                stream.write(msg + self.terminator)

            self.flush()
        except Exception:
            self.handleError(record)


class ElapsedFormatter:
    """ElapsedFormatter class."""

    __new_iteration = 1

    def __init__(self) -> None:
        """Instantiate an ElapsedFormatter object."""
        self.start_time = time()
        self.number_of_iterations: int = 1
        self.current_iteration: int = 0
        self._pbar: tqdm.tqdm = None
        self._resetStyle = Style.RESET_ALL
        self._colors = {
            "INFO": [Back.BLUE, Fore.BLUE],
            "DEBUG": [Back.CYAN, Fore.CYAN],
            "WARNING": [Back.YELLOW + Fore.BLACK, Fore.YELLOW],
            "ERROR": [Back.RED, Fore.RED],
            "CRITICAL": [Back.GREEN, Fore.GREEN],
        }

        colorama.init()

        # saves amount of errors / warnings
        self._loglibrary: dict = {"ERROR": 0, "WARNING": 0}

    @property
    def pbar(self) -> None | tqdm.tqdm | tqdm.std.tqdm:
        """Progress bar."""
        return self._pbar

    @pbar.setter
    def pbar(self, pbar: tqdm.std.tqdm | tqdm.tqdm | None) -> None:
        """Set progress bar."""
        if isinstance(pbar, (tqdm.std.tqdm, type(None))):
            self._pbar = pbar
        else:
            raise TypeError

    def format(self, record: LogRecord) -> str:
        """Format log record."""
        if self._intro:
            return self.__format_intro(record)
        if self.__new_iteration > 0:
            return self.__format_header(record)
        if self.__new_iteration == -1:
            return self.__format_footer(record)
        return self.__format_message(record)

    def __format_intro(self, record: LogRecord) -> str:
        return f"{record.getMessage()}"

    def __format_header(self, record: LogRecord) -> str:
        self.__new_iteration -= 1
        message = record.getMessage()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"╔═════╣ {self._resetStyle}{current_time} {message}{self._resetStyle}"

    def __format_footer(self, record: LogRecord) -> str:
        self.__new_iteration -= 1
        elapsed_seconds = record.created - self.start_time
        return f"╚═════╣ {self._resetStyle}Task finished in {elapsed_seconds:.2f}sec{self._resetStyle}"

    def __format_message(self, record: LogRecord) -> str:
        elapsed_seconds = record.created - self.start_time
        color = self._colors

        level = record.levelname
        message = record.getMessage()

        # Counter of errors, warnings
        if level in self._loglibrary:
            self._loglibrary[level] += 1

        return (
            f"║  {color[level][0]} {level:>7} "
            f"{self._resetStyle}{color[level][1]}{self._resetStyle} T+ {elapsed_seconds:.2f}s {message}"
        )

    def __reset(self) -> None:
        self.start_time = time()

    def start_new_iteration(self, pbar: tqdm.tqdm | None = None) -> None:
        """Start a new iteration with a progress bar."""
        self.current_iteration += 1
        self.new_task()
        self.pbar = pbar

    def new_task(self) -> None:
        """Reset ElapsedTimeFormatter."""
        self.__new_iteration = 1
        self.__reset()

    def finish_task(self) -> None:
        """Finish task."""
        self.__new_iteration = -1

    def set_number_of_iterations(self, n: int) -> None:
        """Set numbber of iterations."""
        if n < 1:
            err_msg = "Total number of iterations should be higher than zero"
            raise ValueError(err_msg)
        self.number_of_iterations = n

    def set_intro(self, flag: bool = True) -> None:  # noqa: FBT001, FBT002
        """Indicate intro section for formatter."""
        self._intro = flag

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time
        return elapsed_time.total_seconds()


class ElapsedFileFormatter(ElapsedFormatter):
    """Elapsed file formatter class."""

    def __init__(self) -> None:
        """Instantiate an ElapsedFileFormatter object."""
        super().__init__()
        self._resetStyle = ""
        self._colors = {
            "INFO": ["", ""],
            "DEBUG": ["", ""],
            "WARNING": ["", ""],
            "ERROR": ["", ""],
            "RESET": "",
        }


class FM2ProfBase:
    """Base class for FM2PROF types.

    Implements methods for logging, project specific parameters
    """

    __logger = None
    __iniFile = None
    __url__ = "https://deltares.github.io/Fm2Prof/"
    __contact__ = "koen.berends@deltares.nl"
    __authors__ = "Koen Berends, Asako Fujisaki, Carles Soriano Perez, Ilia Awakimjan"
    __copyright__ = "Copyright 2016-2020, University of Twente & Deltares"
    __license__ = "LPGL"

    def __init__(self, logger: Logger | None = None, inifile: IniFile | None = None) -> None:
        """Instatiate a FM2ProfBase object.

        Args:
        ----
            logger (Logger | None, optional): Logger . Defaults to None.
            inifile (IniFile | None, optional): IniFile instance. Defaults to None.

        """
        if logger:
            self.set_logger(logger)
        if inifile:
            self.set_inifile(inifile)

    @staticmethod
    def create_logger() -> Logger:
        """Create logger instance."""
        # Create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # create formatter
        logger.__logformatter = ElapsedFormatter()  # noqa: SLF001
        logger._Filelogformatter = ElapsedFileFormatter()  # noqa: SLF001

        # create console handler
        if TqdmLoggingHandler not in map(type, logger.handlers):
            ch = TqdmLoggingHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(logger.__logformatter)  # noqa: SLF001
            logger.addHandler(ch)

        return logger

    def get_logger(self) -> Logger:
        """Use this method to return logger object."""
        return self.__logger

    def set_logger(self, logger: Logger) -> None:
        """Use to set logger.

        Args:
        ----
            logger (Logger): Logger instance

        """
        if not isinstance(logger, Logger):
            err_msg = "logger should be instance of Logger class"
            raise TypeError(err_msg)
        self.__logger = logger

    def set_logger_message(
        self,
        err_mssg: str = "",
        level: str = "info",
        *,
        header: bool = False,
    ) -> None:
        """Set message to logger if this is set.

        Args:
        ----
            err_mssg (str, optional): Error message to log. Defaults to "".
            level (str, optional): Log level. Defaults to "info".
            header (bool, optional): Set error message as header. Defaults to False.

        """
        if not self.__logger:
            return

        if header:
            self.get_logformatter().set_intro(True)
            self.get_logger()._Filelogformatter.set_intro(True)  # noqa: SLF001
        else:
            self.get_logformatter().set_intro(False)
            self.get_logger()._Filelogformatter.set_intro(False)  # noqa: SLF001

        if level.lower() not in ["info", "debug", "warning", "error", "critical"]:
            err_msg = f"{level.lower()} is not valid logging level."
            raise ValueError(err_msg)

        if level.lower() == "info":
            self.__logger.info(err_mssg)
        elif level.lower() == "debug":
            self.__logger.debug(err_mssg)
        elif level.lower() == "warning":
            self.__logger.warning(err_mssg)
        elif level.lower() == "error":
            self.__logger.error(err_mssg)
        elif level.lower() in ["succes", "critical"]:
            self.__logger.critical(err_mssg)

    def start_new_log_task(
        self,
        task_name: str = "NOT DEFINED",
        pbar: tqdm.tqdm = None,
    ) -> None:
        """Use this method to start a new task. Will reset the internal clock.

        :param task_name: task name, will be displayed in log message
        """
        self.get_logformatter().start_new_iteration(pbar=pbar)
        self.get_filelogformatter().start_new_iteration(pbar=pbar)
        self.set_logger_message(f"Starting new task: {task_name}")

    def finish_log_task(self) -> None:
        """Use this method to finish task.

        :param task_name: task name, will be displayed in log message
        """
        self.get_logformatter().finish_task()
        self.set_logger_message()
        self.pbar = None

    def get_logformatter(self) -> ElapsedFormatter:
        """Return log formatter."""
        return self.get_logger().__logformatter  # noqa: SLF001

    def get_filelogformatter(self) -> ElapsedFormatter:
        """Return  file log formatter."""
        return self.get_logger()._Filelogformatter  # noqa: SLF001

    def set_logfile(self, output_dir: str | Path, filename: str = "fm2prof.log") -> None:
        """Set log file.

        Args:
        ----
            output_dir (str): _description_
            filename (str, optional): _description_. Defaults to "fm2prof.log".

        """
        # create file handler
        if not output_dir:
            err_msg = "output_dir is required."
            raise ValueError(err_msg)
        fh = logging.FileHandler(Path(output_dir).joinpath(filename), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.get_logger()._Filelogformatter)  # noqa: SLF001
        self.__logger.addHandler(fh)

    def set_inifile(self, inifile: IniFile = None) -> None:
        """Use this method to set configuration file object.

        For loading from file, use ``load_inifile`` instead

        Args:
        ----
            inifile (IniFile): inifile object. Obtain using e.g. ``get_inifile``.

        """
        self.__iniFile = inifile

    def get_inifile(self) -> IniFile:
        """Get the inifile object."""
        return self.__iniFile


class FrictionTable:
    """Container for friction table."""

    def __init__(self, level: np.ndarray, friction: np.ndarray) -> None:
        """Instantiate a FrictionTable object."""
        if self._validate_input(level, friction):
            self.level = level
            self.friction = friction

    def interpolate(self, new_z: np.ndarray) -> None:
        """Interpolate friction.

        Args:
            new_z (np.ndarray): _description_
        """
        self.friction = np.interp(new_z, self.level, self.friction)
        self.level = new_z

    @staticmethod
    def _validate_input(level: np.ndarray, friction: np.ndarray) -> bool:
        if not isinstance(level, np.ndarray):
            err_msg = f"level argument not of type {np.ndarray}."
            raise TypeError(err_msg)
        if not isinstance(friction, np.ndarray):
            err_msg = f"friction argument not of type {np.ndarray}."
        if level.shape != friction.shape:
            err_msg = "level and friction arrays should have the same shape."
            raise ValueError(err_msg)
        return True
