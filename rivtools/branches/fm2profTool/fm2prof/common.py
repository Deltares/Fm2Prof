# -*- coding: utf-8 -*-
"""
Base classes and data containers
"""

# Imports from standard library
import os
import logging
from logging import Logger, LogRecord
import time
from datetime import datetime
from typing import Mapping, AnyStr

# Import from dependencies
import numpy as np 
import colorama
from colorama import Fore, Back, Style

# Import from package
from fm2prof import IniFile

class ElapsedFormatter:
    """
    Logger formatting class
    """
    __new_iteration = True
    __intro = True
    __colors = {'INFO': [Back.BLUE, Fore.BLUE],
                'DEBUG': [Back.CYAN, Fore.CYAN],
                'WARNING': [Back.YELLOW + Fore.BLACK, Fore.YELLOW],
                'ERROR': [Back.RED, Fore.RED],
                'RESET': Style.RESET_ALL}

    def __init__(self):
        self.start_time = time.time()
        self.number_of_iterations = 1
        self.current_iteration = 0
        colorama.init()

    def format(self, record):
        if self.__new_iteration:
            return self.__format_header(record)
        elif self.__intro:
            return self.__format_intro(record)
        else:
            return self.__format_message(record)

    def __current_time(self) -> AnyStr:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def __format_intro(self, record:LogRecord) -> AnyStr:
        level = record.levelname
        color = self.__colors[level]
        reset = self.__colors['RESET']
        return f"{color[0]}{self.__current_time()} {level:>7} {reset}{color[1]}{reset} {record.getMessage()}" 

    def __format_header(self, record:LogRecord) -> AnyStr:
        self.__new_iteration = False
        elapsed_seconds = record.created - self.start_time
        return "{now} {level:>7} :: {progress:4.0f}% :: {message} ({file})".format(
                now=self.__current_time(),
                level=record.levelname,
                elapsed=elapsed_seconds,
                message=record.getMessage(),
                file=record.filename,
                progress=100*self.current_iteration/self.number_of_iterations)

    def __format_message(self, record) -> AnyStr:
        level = record.levelname
        color = self.__colors[level]
        elapsed_seconds = record.created - self.start_time
        return "{color}{now} {level:>7} {reset}{color2}{reset} {progress:4.0f}% T+ {elapsed:.2f}s {message} ({file})".format(
                color=color[0],
                color2=color[1],
                now=self.__current_time(),
                level=level,
                reset=self.__colors['RESET'],
                elapsed=elapsed_seconds,
                message=record.getMessage(),
                file=record.filename,
                progress=100*self.current_iteration/self.number_of_iterations)

    def __reset(self):
        self.start_time = time.time()

    def start_new_iteration(self):
        self.current_iteration += 1
        self.next_step()

    def next_step(self):
        self.__new_iteration = True
        self.__reset()

    def set_number_of_iterations(self, n):
        assert n > 0, 'Total number of iterations should be higher than zero'
        self.number_of_iterations = n

    def set_intro(self, flag: bool=True):
        self.__intro=flag


class FM2ProfBase:
    """
    Base class for FM2PROF types. Implements methods for logging, project specific parameters
    """
    __logger = None
    __iniFile = None
    __version__ = 1.4
    __contact__ = "koen.berends@deltares.nl"
    __authors__ = "Koen Berends, Asako Fujisaki, Carles Soriano Perez, Ilia Awakimjan"
    __copyright__ = "Copyright 2016-2020, University of Twente & Deltares"
    __license__ = "LPGL"

    def __init__(self, logger:Logger = None, inifile: IniFile=None):
        if logger:
            self.set_logger(logger)
        if inifile:
            self.set_inifile(inifile)

    def _create_logger(self):
        # Create logger
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)

        # create formatter
        self.__logformatter = ElapsedFormatter()

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.__logformatter)
        self.__logger.addHandler(ch)

    def get_logger(self) -> Logger:
        """ Use this method to return logger object """
        return self.__logger

    def set_logger(self, logger:Logger) -> None:
        """ 
        Use to set logger

        Parameters:
            logger (Logger): Logger instance
        """
        assert isinstance(logger, Logger), '' + \
            'logger should be instance of Logger class'
        self.__logger = logger

    def set_logger_message(self, err_mssg: str, level: str='info')->None:
        """Sets message to logger if this is set.

        Arguments:
            err_mssg {str} -- Error message to send to logger.
        """
        if not self.__logger:
            return

        if level.lower() not in [
                'info', 'debug', 'warning', 'error', 'critical']:
            self.__logger.error(
                "{} is not valid logging level.".format(level.lower()))

        if level.lower() == 'info':
            self.__logger.info(err_mssg)
        elif level.lower() == 'debug':
            self.__logger.debug(err_mssg)
        elif level.lower() == 'warning':
            self.__logger.warning(err_mssg)
        elif level.lower() == 'error':
            self.__logger.error(err_mssg)
        elif level.lower() == 'critical':
            self.__logger.critical(err_mssg)

    def start_new_log_task(self, task_name: str="NOT DEFINED") -> None:
        """ 
        Use this method to start a new task. Will reset the internal clock. 

        :param task_name: task name, will be displayed in log message
        """
        self.get_logformatter().next_step()
        self.set_logger_message(f"Starting new task: {task_name}")

    def get_logformatter(self) -> ElapsedFormatter:
        """ Returns formatter """
        return self.__logformatter

    def set_logfile(self, output_dir: str, filename: str='fm2prof.log') -> None:
        # create file handler
        fh = logging.FileHandler(os.path.join(output_dir, filename), encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.__logformatter)
        self.__logger.addHandler(fh)

    def set_inifile(self, iniFilePath: str):
        """ Use this method to set configuration file """
        self.__iniFile = IniFile.IniFile(iniFilePath)

    def get_inifile(self) -> IniFile:
        """"Use this method to get the inifile object """
        return self.__iniFile


class FmModelData:
    """
    Container for FmModelData
    """
    time_dependent_data = None
    time_independent_data = None
    edge_data = None
    node_coordinates = None
    css_data_list = None

    def __init__(self, arg_list: list):
        if not arg_list:
            raise Exception('FM model data was not read correctly.')
        if len(arg_list) != 5:
            raise Exception(
                'Fm model data expects 5 arguments but only ' +
                '{} were given'.format(len(arg_list)))
        
        (self.time_dependent_data,
            self.time_independent_data,
            self.edge_data,
            self.node_coordinates,
            css_data_dictionary) = arg_list
        
        self.css_data_list = self.get_ordered_css_list(css_data_dictionary)

    @staticmethod
    def get_ordered_css_list(css_data_dict: Mapping[str, str]):
        """Returns an ordered list where every element
        represents a Cross Section structure

        Arguments:
            css_data_dict {Mapping[str,str]} -- Dictionary ordered by the keys

        Returns:
            {list} -- List where every element contains a dictionary
            to create a Cross Section.
        """
        if not css_data_dict or not isinstance(css_data_dict, dict):
            return []

        number_of_css = len(css_data_dict[next(iter(css_data_dict))])
        css_dict_keys = css_data_dict.keys()
        css_dict_values = css_data_dict.values()
        css_data_list = [
            dict(zip(
                css_dict_keys,
                [value[idx]
                    for value in css_dict_values if idx < len(value)]))
            for idx in range(number_of_css)]
        return css_data_list


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
        assert level.shape==friction.shape

        return True