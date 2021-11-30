# -*- coding: utf-8 -*-
"""
Base classes and data containers
"""

import logging

# Imports from standard library
import os
import time
from datetime import datetime
from logging import Logger, LogRecord
from typing import AnyStr, Mapping

import colorama

# Import from dependencies
import numpy as np
from colorama import Back, Fore, Style

# Import from package
# none

IniFile = "fm2prof.IniFile.IniFile"


class ElapsedFormatter:
    """
    Logger formatting class
    """

    def __init__(self):
        self.start_time = time.time()
        self.number_of_iterations = 1
        self.current_iteration = 0

        self._new_iteration = False
        self._intro = False
        self._colors = {
            "INFO": [Back.BLUE, Fore.BLUE],
            "DEBUG": [Back.CYAN + Fore.BLACK, Fore.CYAN + Back.BLACK],
            "WARNING": [Back.YELLOW + Fore.BLACK, Fore.YELLOW],
            "ERROR": [Back.RED, Fore.RED],
            "RESET": Style.RESET_ALL,
        }

        self._loglibrary = {
            "INFO": 0,
            "DEBUG": 0,
            "WARNING": 0,
            "ERROR": 0,
        }  # used to store the number of messages

        colorama.init()

    def format(self, record):
        if self._intro:
            return self.__format_intro(record)
        elif self._new_iteration:
            return self.__format_header(record)
        else:
            return self.__format_message(record)

    def get_elapsed_time(self, current_time):
        return current_time - self.start_time

    def __current_time(self) -> AnyStr:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def __format_intro(self, record: LogRecord) -> AnyStr:
        level = record.levelname
        color = self._colors[level]
        reset = self._colors["RESET"]
        return f"{record.getMessage()}"

    def __format_header(self, record: LogRecord) -> AnyStr:
        # Only one header
        self._new_iteration = False
        color = self._colors["DEBUG"][0]
        reset = self._colors["RESET"]
        return f"{color}{self.__current_time()} {record.getMessage()} {reset}"

    def __format_message(self, record) -> AnyStr:
        level = record.levelname
        color = self._colors[level]
        elapsed_seconds = self.get_elapsed_time(record.created)

        # log the number of warnings, errors
        self._loglibrary[level] += 1

        return "{color}{now} {level:>7} {reset}{color2}{reset} {progress:4.0f}% T+ {elapsed:.2f}s {message}".format(
            color=color[0],
            color2=color[1],
            now=self.__current_time(),
            level=level,
            reset=self._colors["RESET"],
            elapsed=elapsed_seconds,
            message=record.getMessage(),
            progress=100 * self.current_iteration / self.number_of_iterations,
        )

    def __reset_time(self):
        self.start_time = time.time()

    def start_new_iteration(self):
        """
        Use this method to print a header
        """
        self.current_iteration += 1
        self._next_step()

    def _next_step(self):
        self._new_iteration = True
        self.__reset_time()

    def set_number_of_iterations(self, n):
        assert n > 0, "Total number of iterations should be higher than zero"
        self.number_of_iterations = n

    def set_intro(self, flag: bool = True):
        self._intro = flag


class ElapsedFileFormatter(ElapsedFormatter):
    def __init__(self):
        super().__init__()

        self._colors = {
            "INFO": ["", ""],
            "DEBUG": ["", ""],
            "WARNING": ["", ""],
            "ERROR": ["", ""],
            "RESET": "",
        }


class FM2ProfBase:
    """
    Base class for FM2PROF types. Implements methods for logging, project specific parameters
    """

    __logger = None
    __iniFile = None
    __version__ = "1.4.4"
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
        if logging.StreamHandler not in map(type, self.__logger.handlers):
            ch = logging.StreamHandler()
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
        self, err_mssg: str, level: str = "info", header: bool = False
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
        elif level.lower() == "critical":
            self.__logger.critical(err_mssg)

    def start_new_log_task(self, task_name: str = "NOT DEFINED") -> None:
        """
        Use this method to start a new task. Will reset the internal clock.

        :param task_name: task name, will be displayed in log message
        """
        self.get_logformatter().start_new_iteration()
        self.set_logger_message(f"Starting new task: {task_name}")

    def get_logformatter(self) -> ElapsedFormatter:
        """Returns formatter"""
        return self.get_logger().__logformatter

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
