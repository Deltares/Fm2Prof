# -*- coding: utf-8 -*-
"""
Base classes and data containers
"""

import logging

# Imports from standard library
import os
from datetime import datetime
from logging import Logger, LogRecord
from time import time
from typing import AnyStr, Mapping

import colorama

# Import from dependencies
import numpy as np
import tqdm
from colorama import Back, Fore, Style

# Import from package
# none

IniFile = "fm2prof.IniFile.IniFile"


class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.formatter.pbar:
                self.formatter.pbar.write(msg)
            else:
                stream = self.stream
                stream.write(msg + self.terminator)

            self.flush()
        except Exception as e:
            self.handleError(record)


class ElapsedFormatter:
    __new_iteration = 1

    def __init__(self):
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
    def pbar(self):
        return self._pbar

    @pbar.setter
    def pbar(self, pbar):
        if isinstance(pbar, (tqdm.std.tqdm, type(None))):
            self._pbar = pbar
        else:
            raise ValueError

    def format(self, record):
        if self._intro:
            return self.__format_intro(record)
        if self.__new_iteration > 0:
            return self.__format_header(record)
        if self.__new_iteration == -1:
            return self.__format_footer(record)
        else:
            return self.__format_message(record)

    def __format_intro(self, record: LogRecord):
        return f"{record.getMessage()}"

    def __format_header(self, record: LogRecord):
        """Formats the header of a new task"""

        self.__new_iteration -= 1
        message = record.getMessage()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"╔═════╣ {self._resetStyle}{current_time} {message}{self._resetStyle}"

    def __format_footer(self, record: LogRecord):
        self.__new_iteration -= 1
        elapsed_seconds = record.created - self.start_time
        message = record.getMessage()
        return f"╚═════╣ {self._resetStyle}Task finished in {elapsed_seconds:.2f}sec{self._resetStyle}"

    def __format_message(self, record: LogRecord):
        elapsed_seconds = record.created - self.start_time
        color = self._colors

        level = record.levelname
        message = record.getMessage()

        # Counter of errors, warnings
        if level in self._loglibrary:
            self._loglibrary[level] += 1

        formatted_string = (
            f"║  {color[level][0]} {level:>7} "
            + f"{self._resetStyle}{color[level][1]}{self._resetStyle} T+ {elapsed_seconds:.2f}s {message}"
        )

        return formatted_string

    def __reset(self):
        self.start_time = time()

    def start_new_iteration(self, pbar: tqdm.tqdm = None):
        self.current_iteration += 1
        self.new_task()
        self.pbar = pbar

    def new_task(self):
        self.__new_iteration = 1
        self.__reset()

    def finish_task(self):
        self.__new_iteration = -1

    def set_number_of_iterations(self, n):
        assert n > 0, "Total number of iterations should be higher than zero"
        self.number_of_iterations = n

    def set_intro(self, flag: bool = True):
        self._intro = flag

    def get_elapsed_time(self):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        return current_time - self.start_time


class ElapsedFileFormatter(ElapsedFormatter):
    def __init__(self):
        super().__init__()
        self._resetStyle = ""
        self._colors = {
            "INFO": ["", ""],
            "DEBUG": ["", ""],
            "WARNING": ["", ""],
            "ERROR": ["", ""],
            "RESET": "",
        }

    @property
    def pbar(self):
        return self._pbar

    @pbar.setter
    def pbar(self, pbar):
        self._pbar = None


class FM2ProfBase:
    """
    Base class for FM2PROF types. Implements methods for logging, project specific parameters
    """

    __logger = None
    __iniFile = None
    __url__ = "https://deltares.github.io/Fm2Prof/"
    __contact__ = "koen.berends@deltares.nl"
    __authors__ = "Koen Berends, Asako Fujisaki, Carles Soriano Perez, Ilia Awakimjan"
    __copyright__ = "Copyright 2016-2020, University of Twente & Deltares"
    __license__ = "LPGL"

    def __init__(self, logger: Logger = None, inifile: IniFile = None):
        if logger:
            self.set_logger(logger)
        if inifile:
            self.set_inifile(inifile)

    def _create_logger(self):
        # Create logger
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)

        # create formatter
        self.__logger.__logformatter = ElapsedFormatter()
        self.__logger._Filelogformatter = ElapsedFileFormatter()

        # create console handler
        if TqdmLoggingHandler not in map(type, self.__logger.handlers):
            ch = TqdmLoggingHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(self.__logger.__logformatter)
            self.__logger.addHandler(ch)

    def get_logger(self) -> Logger:
        """Use this method to return logger object"""
        return self.__logger

    def set_logger(self, logger: Logger) -> None:
        """
        Use to set logger

        Parameters:
            logger (Logger): Logger instance
        """
        assert isinstance(logger, Logger), (
            "" + "logger should be instance of Logger class"
        )
        self.__logger = logger

    def set_logger_message(
        self, err_mssg: str = "", level: str = "info", header: bool = False
    ) -> None:
        """Sets message to logger if this is set.

        Arguments:
            err_mssg {str} -- Error message to send to logger.
        """
        if not self.__logger:
            return

        if header:
            self.get_logformatter().set_intro(True)
            self.get_logger()._Filelogformatter.set_intro(True)
        else:
            self.get_logformatter().set_intro(False)
            self.get_logger()._Filelogformatter.set_intro(False)

        if level.lower() not in ["info", "debug", "warning", "error", "critical"]:
            self.__logger.error("{} is not valid logging level.".format(level.lower()))

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
        self, task_name: str = "NOT DEFINED", pbar: tqdm.tqdm = None
    ) -> None:
        """
        Use this method to start a new task. Will reset the internal clock.

        :param task_name: task name, will be displayed in log message
        """
        self.get_logformatter().start_new_iteration(pbar=pbar)
        self.get_filelogformatter().start_new_iteration(pbar=pbar)
        self.set_logger_message(f"Starting new task: {task_name}")

    def finish_log_task(self) -> None:
        """
        Use this method to finish task.

        :param task_name: task name, will be displayed in log message
        """
        self.get_logformatter().finish_task()
        self.set_logger_message()
        self.pbar = None

    def get_logformatter(self) -> ElapsedFormatter:
        """Returns formatter"""
        return self.get_logger().__logformatter

    def get_filelogformatter(self) -> ElapsedFormatter:
        """Returns formatter"""
        return self.get_logger()._Filelogformatter

    def set_logfile(self, output_dir: str, filename: str = "fm2prof.log") -> None:
        # create file handler
        fh = logging.FileHandler(os.path.join(output_dir, filename), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.get_logger()._Filelogformatter)
        self.__logger.addHandler(fh)

    def set_inifile(self, inifile: IniFile = None):
        """
        Use this method to set configuration file object.

        For loading from file, use ``load_inifile`` instead

        Parameters:
            inifile (IniFile): inifile object. Obtain using e.g. ``get_inifile``.
        """
        self.__iniFile = inifile

    def get_inifile(self) -> IniFile:
        """ "Use this method to get the inifile object"""
        return self.__iniFile


class FrictionTable:
    """
    Container for friction table
    """

    def __init__(self, level, friction):
        if self._validate_input(level, friction):
            self.level = level
            self.friction = friction

    def interpolate(self, new_z):
        self.friction = np.interp(new_z, self.level, self.friction)
        self.level = new_z

    @staticmethod
    def _validate_input(level, friction):
        assert isinstance(level, np.ndarray)
        assert isinstance(friction, np.ndarray)
        assert level.shape == friction.shape

        return True
