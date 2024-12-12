"""Module contains the IniFile class, which handles the main configuration file."""

import configparser
import inspect
import io
import json
import os
from logging import Logger
from pathlib import Path
from pydoc import locate
from typing import Any, Dict, Generator, List, Mapping, Tuple, Type, Union

from fm2prof.common import FM2ProfBase


class InvalidConfigurationFileError(Exception):
    """Raised when config file is not up to snot."""


class ImportBoolType:
    """Custom type to parse booleans."""

    def __new__(cls, value: str) -> bool:  # noqa: D102
        return value.lower().strip() == "true"


class ImportListType:
    """Custom type to parse list of comma separated ints."""

    def __new__(cls, value: str) -> list:  # noqa: D102
        return list(map(int, value.strip("[]").split(",")))


class IniFile(FM2ProfBase):
    """Class for utilizing .ini files.

    Provides all functionality to interact with the configuration file, e.g.:
        - reading
        - validating
        - retrieving parameters
        - retrieving files

    """

    _file: Path = None
    __input_files_key = "input"
    __input_parameters_key = "parameters"
    __input_debug_key = "debug"
    __output_key = "output"
    __output_directory_key = "OutputDirectory"
    __ini_keys = {  # noqa:RUF012
        "map_file": "2dmapoutput",
        "css_file": "crosssectionlocationfile",
        "region_file": "regionpolygonfile",
        "section_file": "sectionpolygonfile",
        "export_mapfiles": "exportmapfiles",
        "css_selection": "cssselection",
        "classificationmethod": "classificationmethod",
        "sdfloodplainbase": "sdfloodplainbase",
        "sdstorage": "sdstorage",
        "transitionheight_sd": "transitionheight_sd",
        "number_of_css_points": "number_of_css_points",
        "minimum_width": "minimum_width",
    }

    _output_dir = None
    _input_file_paths = None
    _input_parameters = None

    def __init__(self, file_path: Union[Path, str] = ".", logger: Union[Logger, None] = None) -> None:
        """Initialize the object Ini File.

        File should contain the path locations of all
        parameters needed by the Fm2ProfRunner.

        The configuration file consists of three main sections: input, parameters and output.

            - **input** specifies paths to files. A check will be performed:
                - whether the file exists.
            - **parameters** specified parameters. A check will be performed:
                -  whether the parameters can be typeforced to the expected types
                -  whether the value conforms to allowed values
            - **output** specifies the output directory. A check will be performed:
                - whether the directory can be created

        Args:
            file_path (str): File path where the IniFile is located.
            logger (Logger): logger object to log messages to.

        """
        super().__init__(logger=logger)

        self._ini_template = self._get_template_ini()  # Template to fill defaults from
        self._configuration = self._get_template_ini()  # What will be used

        if file_path is None:
            self.set_logger_message(
                "No ini file given, using default options",
                "warning",
            )
            return
        file_path = Path(file_path)
        if isinstance(file_path, Path):
            self._file = file_path
        if file_path.is_file():
            self.set_logger_message(f"Received ini file: {self._file}", "debug")
            self._read_inifile(file_path)
        elif file_path.is_dir():
            self.set_logger_message(
                "No ini file given, using default options",
                "warning",
            )
        else:
            # User has supplied a file, but the file does not exist. Raise error.
            err_msg = f"The given file path {file_path} could not be found"
            raise OSError(err_msg)

    @property
    def _file_dir(self) -> Path:
        if self._file is not None:
            return self._file.parent
        return Path().cwd()

    @property
    def has_output_directory(self) -> bool:
        """Verifies if the output directory has been set and exists or not.

        Returns:
            True - the output_dir is set and exists.
            False - the output_dir is not set or does not exist.

        """
        if self.get_output_directory() is None:
            self.set_logger_message("No Output Set", "warning")
            return False

        if not self.get_output_directory().exists():
            try:
                self.get_output_directory().mkdir(parents=True)
            except OSError:
                self.set_logger_message(
                    f"The output directory {self.get_output_directory()}, could not be found neither created.",
                    "warning",
                )
                return False

        return True

    def get_output_directory(self) -> Path:
        """Use this method to return the output directory.

        Returns:
            output directory (Path)

        """
        op = self._configuration["sections"]["output"][self.__output_directory_key]["value"]

        return self.get_relative_path(op)

    def set_output_directory(self, name: Union[Path, str]) -> None:
        """Use this method to set the output directory.

        Args:
            name (Union[Path, str]): name of the output directory

        """
        self._configuration["sections"]["output"][self.__output_directory_key]["value"] = self._get_valid_output_dir(
            name
        )

        return self.get_output_directory()

    def get_ini_root(self, dirtype: str = "relative") -> Path:
        """Get root directory of ini file.

        Args:
            dirtype (str, optional): Abosulte or relative path. Defaults to "relative".

        Returns:
            Path: _description_
        """
        if dirtype == "absolute":
            return Path.cwd().joinpath(self._file_dir)
        return self._file_dir

    def get_relative_path(self, path: str) -> Path:
        """Get relative path of the ini root.

        Args:
            path (str): file path.

        Returns:
            Path: relative path to the ini root directory.
        """
        return self.get_ini_root().joinpath(path)

    def get_input_file(self, file_name: str) -> Path:
        """Use this method to retrieve the path to an input file.

        Args:
            file_name (str): _description_

        Returns:
            AnyStr: _description_
        """
        return Path(self._get_from_configuration("input", file_name))

    def get_parameter(self, key: str) -> Union[str, bool, int, float, None]:
        """Use this method to return a parameter value.

        Args:
            key (str): name of parameter.

        Returns:
            Union[str, bool, int, float]: parameter value.
        """
        try:
            return self._get_from_configuration("parameters", key)
        except KeyError:
            pass
        try:
            return self._get_from_configuration("debug", key)
        except KeyError:
            pass
        self.set_logger_message(f"unknown key {key}", "error")
        return None

    def _set_config_value(self, section: str, key: str, value: Any) -> None:  # noqa: ANN401
        """Use this method to set a input files the configuration.

        Args:
            section (str): name of section.
            key (str): name of key.
            value (Any): config value to set.

        """
        ckey = self._get_key_from_case_insensitive_input(section=section, key_in=key)
        if ckey:
            self._configuration["sections"][section][ckey]["value"] = value
            return True
        keys = self._configuration["sections"][section].keys()
        self.set_logger_message(f"Unknown {section}. Available keys: {list(keys)}")
        return False

    def _set_output_directory_no_validation(self, value: Union[str, Path]) -> None:
        """Set the output directory within testing framework only.

        Args:
            value (Union[str, Path]): output directory
        """
        self._configuration["sections"]["output"][self.__output_directory_key]["value"] = value

    def print_configuration(self) -> str:
        """Print the configuration as a string."""
        f = io.StringIO()
        for sectionname, section in self._configuration.get("sections").items():
            f.write(f"[{sectionname}]\n")
            for key, contents in section.items():
                f.write(
                    f"{key:<30}= {contents.get('value')!s:<10}# {contents.get('hint')}\n",
                )
            f.write("\n")
        return f.getvalue()

    def iter_parameters(self) -> Generator[Tuple[str]]:
        """Iterate through the names and values of all parameters."""
        for parameter, content in self._configuration["sections"].get("parameters").items():
            yield (
                parameter,
                content.get("type"),
                content.get("hint"),
                content.get("value"),
            )

    def _get_key_from_case_insensitive_input(self, section: str, key_in: str) -> str:
        section = self._configuration["sections"][section]
        for key in section:
            if key.lower() == key_in.lower():
                return key
        return ""

    def _get_from_configuration(
        self,
        section: str,
        key: str,
    ) -> Union[str, bool, int, float]:
        for parameter, content in self._configuration["sections"].get(section).items():
            if parameter.lower() == key.lower():
                return content.get("value")
        err_msg = f"key {key} not found"
        raise KeyError(err_msg)

    def _get_template_ini(self) -> Dict:
        # Open config file
        path, _ = os.path.split(inspect.getabsfile(IniFile))
        with Path(path).joinpath("configurationfile_template.json").open("r") as f:
            return json.load(f)

    def _read_inifile(self, file_path: str) -> None:
        """Reads the inifile and extract all its parameters for later usage.

        Parameters
        ----------
            file_path {str} -- File path where the IniFile is located

        """
        if not file_path:
            msg = "No ini file was specified and no data could be read."
            self.set_logger_message(msg, "error")
            raise OSError(msg)
        try:
            if not Path(file_path).exists():
                msg = f"The given file path {file_path} could not be found."
                self.set_logger_message(msg, "error")
                raise OSError(msg)
        except TypeError as err:
            if not isinstance(file_path, io.StringIO):
                err_msg = "Unknown file format entered"
                raise TypeError(err_msg) from err
        try:
            supplied_ini = self._get_inifile_params(file_path)
        except Exception as e_info:
            raise Exception(
                f"It was not possible to extract ini parameters from the file {file_path}. Exception thrown: {e_info!s}",
            )

        # Compare supplied with default/expected inifile
        try:
            self._extract_input_files(supplied_ini)
        except Exception:
            self.set_logger_message(
                "Unexpected error reading input files. Check config file",
                "error",
            )
        try:
            self._extract_parameters(supplied_ini, self.__input_parameters_key)
            self._extract_parameters(supplied_ini, self.__input_debug_key)
        except Exception:
            self.set_logger_message(
                "Unexpected error reading (debug) parameters. Check config file",
                "error",
            )
        try:
            self._extract_output_dir(supplied_ini)
        except Exception:
            self.set_logger_message(
                "Unexpected error output parameters. Check config file",
                "error",
            )

    def _extract_parameters(self, supplied_ini: Mapping[str, list], section: str) -> None:
        """Extract InputParameters and convert values either integer or float from string.

        Args:
            supplied_ini (Mapping[str, list]): Mapping of ini config parameters
            section (str): name of section
        """
        try:
            inputsection = supplied_ini.get(section)
        except KeyError as err:
            raise InvalidConfigurationFileError from err

        for key, value in inputsection.items():
            key_default, key_type = self._get_key_from_template(section, key)
            try:
                parsed_value = key_type(value)
                self._set_config_value("parameters", key_default, parsed_value)
            except ValueError:
                self.set_logger_message(
                    f"{key} could not be cast as {key_type}",
                    "debug",
                )
            except KeyError:
                pass

    def _extract_input_files(self, supplied_ini: Mapping[str, list]) -> None:
        """Extract and validate input files.

        Args:
            supplied_ini (Mapping[str, list]):  Mapping of ini config parameters
        """
        try:
            inputsection = supplied_ini.get(self.__input_files_key)
        except KeyError as err:
            raise InvalidConfigurationFileError from err

        for key, input_file in inputsection.items():
            key_default, _ = self._get_key_from_template("input", key)
            if key_default is None:
                continue

            input_file_path = self._file_dir.joinpath(input_file)

            if input_file_path.is_file():
                self._set_config_value("input", key_default, input_file_path)
                continue

            if key_default in ("2DMapOutput", "CrossSectionLocationFile"):
                err_msg = f"Could not find input file: {key_default}"
                self.set_logger_message(
                    err_msg,
                    "error",
                )
                raise FileNotFoundError(err_msg)
            self.set_logger_message(
                f"Could not find optional input file for {key_default}, skipping",
                "warning",
            )

    def _get_key_from_template(self, section: str, key: str) -> List[Union[str, Type]]:
        """Return list of lower case keys from default configuration files."""
        sectiondict = self._ini_template.get("sections").get(section)
        for entry in sectiondict:
            if key.lower() == entry.lower():
                return (entry, locate(sectiondict[entry].get("type")))
        # If not returned by now, key must be unknown
        self.set_logger_message(f"{key} is not a known key", "warning")
        return [None, KeyError]

    def _extract_output_dir(self, supplied_ini: Mapping[str, list]) -> None:
        """Extract and validates output directory.

        Args:
            supplied_ini (Mapping[str, list]): Mapping of ini config parameters

        """
        try:
            outputsection = supplied_ini.get(self.__output_key)
        except KeyError as err:
            raise InvalidConfigurationFileError from err

        for key, value in outputsection.items():
            if key.lower() == self.__output_directory_key.lower():
                self.set_output_directory(value)
            else:
                self.set_logger_message(
                    f"Unknown key {key} found in configuration file",
                    "warning",
                )

    def _get_valid_output_dir(self, output_dir: str) -> Path:
        """Get a normalized output directory path. Creates it if not yet exists.

        Args:
            output_dir (str): Relative path to the configuration file.

        Returns:
            _Path: Valid output directory path.
        """
        output_dir = Path(output_dir)
        if output_dir.exists():
            return output_dir
        output_dir.mkdir()
        return output_dir

    @property
    def _output_dir(self) -> Union[str, None]:
        try:
            return self._configuration["sections"]["output"][self.__output_directory_key]["value"]
        except KeyError:
            return None

    @staticmethod
    def _get_inifile_params(file_path: str) -> Dict:
        """Extracts the parameters from an ini file.

        Args:
            file_path (str): Ini file location.

        Returns:
            Dict: config file parameters
        """
        ini_file_params = {}
        comment_delimter = "#"

        # Create config parser (external class)
        config = configparser.ConfigParser()
        if isinstance(file_path, io.StringIO):
            config.read_file(file_path)
        else:
            with Path(file_path).open("r") as f:
                config.read_file(f)

        # Extract all sections and options
        for section in config.sections():
            ini_file_params[section.lower()] = {}
            for option in config.options(section):
                ini_file_params[section.lower()][option.lower()] = (
                    config.get(section, option).split(comment_delimter)[0].strip()
                )

        return ini_file_params
