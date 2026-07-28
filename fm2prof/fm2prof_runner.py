"""FM2PROF Runner Module.

This module provides the main execution engine for FM2PROF, a tool for converting
2D dflowFM model results to 1D cross-sections for hydraulic modelling.

FM2PROF (dflowFM to Profile) extracts cross-sectional data from 2D hydrodynamic
model outputs and generates 1D model inputs. The main workflow includes:

1. **Initialisation**: Load configuration files and validate input data
  -. **Data Import**: Read dflowfm map files and cross-section location files
  -. **Classification**: Assign 2D model points to regions and cross-sections
2. **Generation**: Create cross-section geometries and roughness tables
3. **Finalisation**: Write output files in various formats (D-Flow 1D, SOBEK 3, etc.)

Classes:
    InitializationError: Custom exception for initialisation failures.
    Fm2ProfRunner: Main class that orchestrates the FM2PROF workflow.
    Project: Python API wrapper for programmatic access to FM2PROF functionality.

The Project class is auto-imported when using the fm2prof package.

Example:
    Basic usage through the Project API:

    >>> from fm2prof import Project
    >>> project = Project('config.ini')
    >>> project.run()

    Programmatic configuration:

    >>> project = Project()
    >>> project.set_input_file('2DMapOutput', 'model_map.nc')
    >>> project.set_input_file('CrossSectionLocationFile', 'crosssections.csv')
    >>> project.set_output_directory('./output')
    >>> project.run()

Note:
    This module requires FlowFM map files (NetCDF format) and cross-section
    location files as input. The output includes 1D model geometry and
    roughness data suitable for various hydraulic modelling software.

License:
    GPL-3.0-or-later AND LGPL-3.0-or-later
"""

import datetime
import pickle
from collections.abc import Generator
from pathlib import Path

import geojson
import numpy as np
import tqdm
from geojson import Feature, FeatureCollection, Polygon
from scipy.spatial import ConvexHull

from fm2prof import mask_output_file
from fm2prof.common import FM2ProfBase, get_version
from fm2prof.cross_section import CrossSection, CrossSectionHelpers
from fm2prof.data_preprocessing import build_model_data
from fm2prof.export import ExporterFactory
from fm2prof.imports import ModelData
from fm2prof.ini_file import ConfigurationFileError, IniFile, InputFiles
from fm2prof.polygon_file import PolygonError


class InitializationError(Exception):
    """Exception class for initialization errors."""


class Fm2ProfRunner(FM2ProfBase):
    """Main class that executes all functionality."""

    __map_key = "2DMapOutput"
    __css_key = "CrossSectionLocationFile"
    __key_frictionweighingmethod = "FrictionweighingMethod"
    __key_skipmaps = "SkipMaps"

    # Expected output files (used to check if output exists)
    _output_files = [
        "dflow1d/CrossSectionDefinitions.ini",  # D-Flow 1D geometry
    ]

    def __init__(self, ini_file_path: Path | str = "") -> None:
        """Initialize the project and load configuration.

        Args:
        ----
            ini_file_path (Path | str): path to configuration file.

        """
        self.version: str = get_version()
        self.model_data: ModelData = None

        self.set_logger(self.create_logger())

        ini_file_path = Path(ini_file_path)

        self.start_new_log_task("Loading configuration file")
        try:
            self.load_configuration(ini_file_path)
        except (ConfigurationFileError, FileNotFoundError) as e:
            self.set_logger_message(f"Exiting {e}", "error")
            return

        if not self.get_inifile().has_output_directory:
            self.set_logger_message(
                "Output directory must be set in configuration file",
                "error",
            )
            return

        # Add a log file
        self.set_logfile(
            output_dir=self.get_inifile().get_output_directory(),
            filename="fm2prof.log",
        )

        self.finish_log_task()
        # print header to log
        self._print_header()

        # Print configuration to log
        self.set_logger_message(self.get_inifile().print_configuration(), header=True)

    def run(self, *, overwrite: bool = False) -> bool:
        """Execute FM2PROF routines.

        Args:
            overwrite (bool): if True, overwrites existing output. If False, exits if output detected.

        Returns:
            bool: True if run was successful, False if errors occurred.
        """
        if self.get_inifile() is None:
            self.set_logger_message(
                "No ini file was specified: the run cannot go further.",
                "Warning",
            )
            return False

        # Check for already existing output
        if self._output_exists() and not overwrite:
            self.set_logger_message(
                "Output already exists. Use overwrite option if you want to re-run the program",
                "warning",
            )
            return False

        # Run
        success = self._run_inifile()

        if not success:
            self.set_logger_message("Program finished with errors", "warning")
        else:
            self.set_logger_message("Program finished", "info")

        return success

    def load_configuration(self, ini_file_path: Path) -> None:
        """Use this method to load a configuration file from path.

        If no path is given, the default configuration is used.

        Args:
        ----
            ini_file_path (Path | str): path to configuration file

        """
        if not ini_file_path.is_file():
            self.set_logger_message("No ini file path given, using default configuration", "warning")
            ini_file_object = IniFile(logger=self.get_logger())
        else:
            ini_file_object = IniFile(ini_file_path, logger=self.get_logger())
        self.set_inifile(ini_file_object)

    def _print_header(self) -> None:
        header_text = [
            "=" * 80,
            f"FM2PROF version {get_version()}",
            f"Documentation: {self.__url__:>6}",
            f"Authors: {self.__authors__:>6}",
            f"Contact: {self.__contact__:>6}",
            f"License: {self.__license__:>6} license. For more info see LICENSE.txt",
            f"{self.__copyright__:>6}",
            "=" * 80,
            "",
        ]
        for line in header_text:
            self.set_logger_message(line, header=True)

    def _run_inifile(self) -> bool:
        """Execute main program from the configuration file.

        The main steps in the program are:

        1. Initialize fm2prof
        2. Generate cross-sections
        3. Finalization

        """
        # Step 1. Initialise the project
        self.start_new_log_task("Initialising FM2PROF")
        try:
            self._initialise_fm2prof()
        except InitializationError as e:
            self.set_logger_message(f"Initialization failed: {e}", "error")
            return False
        self.finish_log_task()

        # Step 2. Generate cross-sections
        cross_sections = self._generate_cross_section_list()

        # Step 3. Finalise and write output
        self.start_new_log_task("Finalizing")
        self._finalise_fm2prof(cross_sections)
        errors = self._print_log_report()
        self.finish_log_task()

        # Returns true if program finished without errors
        return errors == 0

    def _initialise_fm2prof(self) -> bool:
        """Load data, inifile."""
        ini_file: IniFile = self.get_inifile()
        raise_file_not_found: bool = False

        input_files: InputFiles = ini_file.get_input_files()

        # Check if mandatory input exists
        if not Path(input_files.map_file).is_file():
            self.set_logger_message(
                f"File for {self.__map_key} not found at {input_files.map_file}",
                "error",
            )
            raise_file_not_found = True
        if not Path(input_files.css_file).is_file():
            self.set_logger_message(
                f"File for {self.__css_key} not found at {input_files.css_file}",
                "error",
            )
            raise_file_not_found = True
        if raise_file_not_found:
            raise InitializationError

        # Preprocess the data
        try:
            self.model_data = build_model_data(
                input_files,
                source="dflowfm",
                default_region=ini_file.get_parameter("DefaultRegion"),
                default_section=ini_file.get_parameter("DefaultSection"),
                logger=self.get_logger(),
            )
        except PolygonError as e:
            self.set_logger_message(f"Error during initialisation: {e}", "error")
            raise InitializationError from e

        # Validate config file
        success: bool = self._validate_config_after_initalization()
        if not success:
            self.set_logger_message(
                "Validation of configuration file not successful. Check the log to fix errors.",
                "error",
            )
            raise InitializationError

        ntsteps: int = self.model_data.hydraulics.waterlevel.shape[1]
        nfaces: int = self.model_data.hydraulics.waterlevel.shape[0]
        nedges: int = self.model_data.edges.x.shape[0]
        self.set_logger_message("finished reading FM and cross-sectional data data")
        self.set_logger_message(
            f"Number of: timesteps ({ntsteps}), "
            f"faces ({nfaces}), "
            f"edges ({nedges})",
            level="debug",
        )

        return success

    def _generate_cross_section_list(self) -> list[CrossSection]:
        """Generate cross sections based on the given model_data.

        Returns:
        -------
            (list): List of generated cross sections

        """
        cross_sections = []
        if not self.model_data:
            return cross_sections

        # Preprocess css from model_data so it's easier to handle it.
        css_data_list = self.model_data.css_data_list

        # Set the number of cross-section for progress bar
        css_selection = self._get_css_range(number_of_css=len(css_data_list))
        self.get_logformatter().set_number_of_iterations(len(css_selection) + 1)
        selected_list = np.array(css_data_list)[css_selection]

        # Generate cross-sections one by one
        pbar = tqdm.tqdm(total=len(selected_list))
        for i, css_data in enumerate(selected_list):
            self.start_new_log_task(
                f"{css_data.get('id')}  ({i}/{len(selected_list)})",
                pbar=pbar,
            )
            generated_cross_section = self._generate_cross_section(
                css_data,
                self.model_data,
            )
            if generated_cross_section is not None:
                cross_sections.append(generated_cross_section)
            pbar.update(1)

        return cross_sections

    def _finalise_fm2prof(self, cross_sections: list[CrossSection]) -> None:
        """Write to output, perform checks."""
        self.set_logger_message("Interpolating roughness")
        CrossSectionHelpers().interpolate_friction_across_cross_sections(cross_sections)

        # Export cross sections
        output_dir = self.get_inifile().get_output_directory()
        self.set_logger_message(f"Export model input files to {output_dir}")
        self._write_output(cross_sections, output_dir)

        # Generate debug output
        self._create_debug_output_if_not_exists(output_dir / "debug")

        try:
            export_mapfiles = self.get_inifile().get_parameter("ExportMapFiles")
        except KeyError:
            # If key is missing, do not export files by default.
            # We need a better solution for this (inifile.getparam?.. handle defaults there?)
            export_mapfiles = False
        if export_mapfiles:
            self.set_logger_message(f"Export geojson output to {output_dir}/debug")
            self._generate_geojson_output(output_dir / "debug", cross_sections)

        # Export bounding boxes of cross-section control volumes
        try:
            self._export_envelope(output_dir / "debug", cross_sections)
        except Exception as e_error:
            e_message = str(e_error)
            self.set_logger_message("Error while exporting bounding boxes", "error")
            self.set_logger_message(e_message, "error")

    def _validate_config_after_initalization(self) -> bool:
        """Perform validation checks on config file.

        Returns True if all checks succesful, False if check fails.
        """
        success: bool = True

        self.set_logger_message("Validating settings", "Info")

        # Check if skipmaps is lower than maximum amount of maps
        nsteps: int = self.model_data.hydraulics.waterlevel.shape[1]
        skipmap: int = self.get_inifile().get_parameter(self.__key_skipmaps)

        if skipmap >= nsteps:
            self.set_logger_message(
                f"""You are attempting to skip more than  available timesteps.
                ({self.__key_skipmaps} = {skipmap}, available maps in output file:
                 {nsteps}). Modify the value of {self.__key_skipmaps}
                in your configuration file to fix this error.""",
                level="error",
            )
            success = False
        elif skipmap > nsteps / 2:
            self.set_logger_message(
                f"""You are skipping more than half of available timesteps.
                    ({self.__key_skipmaps} = {skipmap}, available maps in output file: {nsteps})""",
                level="warning",
            )

        # Check if edge/face data is available
        if (
            self.model_data.edges.edge_faces is None
            and self.get_inifile().get_parameter(self.__key_frictionweighingmethod) == 1
        ):
            self.set_logger_message(
                "Friction weighing set to 1 (area-weighted average"
                "but FM map file does contain the *edge_faces* keyword."
                "Area weighting is not possible. Defaulting to simple unweighted"
                "averaging",
                level="warning",
            )

        return success

    def _create_debug_output_if_not_exists(self, output_dir: Path) -> None:
        """Create debug output directory if it does not exist."""
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

    def _export_envelope(
        self,
        output_dir: Path | str,
        cross_sections: list[CrossSection],
    ) -> None:
        """Export envelopes around cross-sections."""
        css_hulls = []
        for css in cross_sections:
            pointlist = np.array(
                [
                    point["geometry"]["coordinates"]
                    for point in css.get_point_list("face")
                ],
            )
            # construct envelope
            try:
                hull = ConvexHull(pointlist)
                css_hulls.append(
                    Feature(
                        properties={"name": css.name},
                        geometry=Polygon([list(map(tuple, pointlist[hull.vertices]))]),
                    ),
                )
            except IndexError:
                self.set_logger_message(f"No Hull Exported For {css.name}")
        with Path(output_dir).joinpath("cross_section_volumes.geojson").open("w") as f:
            geojson.dump(FeatureCollection(css_hulls), f, indent=2)

    def _generate_geojson_output(self, output_dir: str, cross_sections: list) -> None:
        """Generate geojson file based on cross sections.

        Args:
        ----
            output_dir (str): Output directory path.
            cross_sections (list): List of Cross Sections.

        """
        for pointtype in ["face", "edge"]:
            output_file_path = Path(output_dir) / f"{pointtype}_output.geojson"
            try:
                node_points = [
                    node_point
                    for cs in cross_sections
                    for node_point in cs.get_point_list(pointtype)
                ]
                self.set_logger_message(
                    "Collected points, dumping to file",
                    level="debug",
                )
                mask_output_file.write_mask_output_file(output_file_path, node_points)
                self.set_logger_message("Done", level="debug")
            except Exception as e_info:
                self.set_logger_message(
                    ("Error while generation .geojson file,"
                     f"at {output_file_path}"
                     f"Reason: {e_info!s}"),
                    level="error",
                )

    def _get_css_range(self, number_of_css: int) -> np.array:
        """Parse the CssSelection keyword from the inifile."""
        css_selection = self.get_inifile().get_parameter("CssSelection")
        return (
            np.arange(0, number_of_css)
            if not css_selection
            else np.array(css_selection)
        )

    def _generate_cross_section(
        self,
        css_data: dict,
        model_data: ModelData,
    ) -> CrossSection:
        """Generate a cross section and configures its values based.

        on the input parameter dictionary

        Args:
        ----
            css_data (dict): Dictionary of data for the current cross section.
            model_data (FmModelData): Data to assign to the new cross section

        Raises:
        ------
            Exception: If no css_data is given.
            Exception: If no input_param_dict is given.
            Exception: If no model_data is given.

        Returns:
        -------
            (CrossSection): New Cross Section

        """
        if css_data is None:
            err_msg = "No data was given to create a Cross Section"
            raise ValueError(err_msg)

        css_name = css_data.get("id")
        if not css_name:
            css_name = "new_cross_section"

        if model_data is None:
            err_msg = f"No FM data given for new cross section {css_name}"
            raise ValueError(err_msg)

        # Create cross section
        created_css = self._create_new_cross_section(css_data=css_data)

        if created_css is None:
            self.set_logger_message(
                f"No Cross-section could be generated for {css_name}",
                "error",
            )
            return None
        if created_css.get_number_of_faces() < 10:  # noqa: PLR2004
            self.set_logger_message(
                "There are too few 2D points in control volume to construct cross-section",
                "error",
            )
            return None

        self.set_logger_message("Initiated new cross-section", "info")
        self._build_cross_section_geometry(cross_section=created_css)
        self._build_cross_section_roughness(cross_section=created_css)

        # if self.get_inifile().get_parameter('ExportMapFiles'):
        created_css.set_face_output_list()
        created_css.set_edge_output_list()

        if created_css is not None:
            self.finish_log_task()
        return created_css

    def _build_cross_section_geometry(
        self,
        cross_section: CrossSection,
    ) -> CrossSection:
        """Manage the options of building the cross-section geometry.

        Args:
        ----
            cross_section (CrossSection): Given Cross Section.

        """
        if cross_section is None:
            err_msg = "Cross section cannot be none."
            raise ValueError(err_msg)

        # Build cross-section
        self.set_logger_message("Start building geometry", "debug")
        cross_section.build_geometry()

        # 2D Volume Correction (SummerDike option)
        if self.get_inifile().get_parameter("SDCorrection"):
            self.set_logger_message("Starting correction", "debug")
            cross_section = self._perform_2D_volume_correction(cross_section)
        else:
            self.set_logger_message(
                "SD Correction not enable in configuration file, skipping",
                "info",
            )

        # Check if geometry conforms to the requirements of the output formats
        cross_section.check_geometry_requirements()

        # Reduce number of points in cross-section
        return self._reduce_css_points(cross_section)

    def _build_cross_section_roughness(
        self,
        cross_section: CrossSection,
    ) -> CrossSection:
        """Build the roughness tables."""
        # Assign roughness
        self.set_logger_message("Starting computing roughness tables", "debug")
        cross_section.assign_roughness()
        self.set_logger_message("Computed roughness", "info")

        # Check if roughness conforms to the requirements of the output formats
        cross_section.check_section_width_requirements()
        return cross_section

    def _create_new_cross_section(self, css_data: dict) -> CrossSection | None:
        """Create a cross section with the given input param dictionary.

        Args:
        ----
            css_data (dict): FM Model data for cross section.

        Returns:
        -------
            (CrossSection): New cross section object.

        """
        # Get id data and id index
        if not css_data:
            return None

        if not css_data.get("id"):
            return None

        if (
            css_data.get("length") is None
            or css_data.get("xy") is None
            or css_data.get("branchid") is None
            or css_data.get("chainage") is None
        ):
            return None

        # Get remainig data
        css_data["model_data"] = self.model_data.get_selection(css_data.get("id"))

        if self.get_inifile().get_parameter("ExportCSSData"):
            output_dir = Path(self.get_inifile().get_output_directory())
            with output_dir.joinpath(f"{css_data.get('id')}.pickle").open("wb") as f:
                pickle.dump(css_data, f)

        return CrossSection(
            logger=self.get_logger(),
            inifile=self.get_inifile(),
            data=css_data,
        )

    def _write_output(self, cross_sections: list, output_dir: Path) -> None:
        """Export all cross sections to the necessary file formats.

        Args:
        ----
            cross_sections (list): List of created cross sections
            output_dir (str): target directory where to export all the cross sections

        """
        if not cross_sections or not output_dir.exists():
            return

        # Export D-Hydro format
        try:
            dhydro_exporter = ExporterFactory.create("dhydro", output_dir=output_dir / "dhydro")
            dhydro_exporter.export_all(cross_sections)
            self.set_logger_message("Successfully exported D-Hydro format files", "info")
        except (ValueError, OSError, KeyError) as e_info:
            self.set_logger_message(
                "An error was produced while exporting files to D-Hydro format,"
                " not all output files might be exported. "
                f"{e_info!s}",
                level="error",
            )

        # Export D-Flow 1D format
        try:
            dflow1d_exporter = ExporterFactory.create("dflow1d", output_dir=output_dir / "dflow1d")
            dflow1d_exporter.export_all(cross_sections)
            self.set_logger_message("Successfully exported D-Flow 1D format files", "info")
        except (ValueError, OSError, KeyError) as e_info:
            self.set_logger_message(
                "An error was produced while exporting files to D-Flow 1D format,"
                " not all output files might be exported. "
                f"{e_info!s}",
                level="error",
            )

        self.set_logger_message("Exported output files, FM2PROF finished")

    def _reduce_css_points(self, cross_section: CrossSection) -> CrossSection:
        """Return a valid value for the number of css points read from ini file.

        Parameters
        ----------
            cross_section (CrossSection)

        Returns:
        -------
            cross_section (CrossSection): modified

        """
        maximum_number_of_css_points = self.get_inifile().get_parameter(
            "MaximumPointsInProfile",
        )

        try:
            cross_section.reduce_points(count_after=maximum_number_of_css_points)
        except Exception as e_error:
            e_message = str(e_error)
            self.set_logger_message(
                (
                    "Exception thrown while trying to reduce the css points. "
                    f"{e_message}"
                ),
                "error",
            )

        return cross_section

    def _get_time_stamp_seconds(self, start_time: datetime) -> float:
        """Return a time stamp with the time difference.

        Args:
        ----
            start_time (datetime): Initial date time

        Returns:
        -------
            (float): difference of time between start and now in seconds

        """
        time_now = datetime.datetime.now()
        time_difference = time_now - start_time
        return time_difference.total_seconds()

    def _perform_2D_volume_correction(self, css: CrossSection) -> CrossSection:  # noqa: N802
        """Calculate a logistic correction term which may be applied in 1D models.

        In 2D, the volume available in a profile can rise rapidly
        while the water level changes little due to compartimentalisation
        of the floodplain. This methods calculates a logistic correction
        term which may be applied in 1D models.

        In SOBEK this option is available as the 'summerdike' options.
        Calculates the Cross Section correction if needed.

        """
        try:
            css.calculate_correction()
            self.set_logger_message("correction finished")
        except Exception as e_error:
            e_message = str(e_error)
            self.set_logger_message(
                (
                    "Exception thrown "
                    "while trying to calculate the correction. "
                    f"{e_message}"
                ),
                "error",
            )
        return css

    def _print_log_report(self) -> int:
        """Print a report of the log with amount of warnings and errors and returns number of errors."""
        ll = self.get_logformatter().get_loglibrary()
        self.set_logger_message(f"Warnings: {ll.get('WARNING')}")
        self.set_logger_message(f"Errors: {ll.get('ERROR')}")

        return ll.get("ERROR")

    def _output_exists(self) -> bool:
        """Check whether output exists."""
        for output_file in self._output_files:
            if (
                self.get_inifile()
                .get_output_directory()
                .joinpath(output_file)
                .is_file()
            ):
                return True
        return False


class Project(Fm2ProfRunner):
    """Provides the python API for running FM2PROF.

    Instantiate by providing the path to a configuration file

    >> Project('/path/to/config.ini')

    """

    def set_parameter(self, name: str, value: str | float) -> None:
        """Use this method to set the value of a parameter.

        Args:
        ----
            name (str): name of the parameter (case insensitive).

            value (str | float): value of the parameter.
            An error will be given if the value has the wrong type (e.g. string if int was expected).

        """
        self.get_inifile().set_parameter(name, value)

    def get_parameter(self, name: str) -> str | float:
        """Use this method to get the value of a parameter.

        Args:
        ----
            name (str): name of the parameter (case insensitive)

        Returns:
        -------
            (str | float): The current value of the parameter

        """
        return self.get_inifile().get_parameter(name)

    def set_input_file(self, name: str, value: str | float) -> None:
        """Use this method to set the path to an input file.

        Args:
        ----
            name: name of the input file in the configuration (case insensitive).

            value: path to the inputfile

        """
        return self.get_inifile().set_input_file(name, value)

    def get_input_file(self, name: str) -> str:
        """Use this method to retrieve the path to an input file.

        Args:
        ----
            name (str): case-insensitive key of the input file (e.g.'2dmapoutput')

        """
        return self.get_inifile().get_input_file(name)

    def set_output_directory(self, path: str | Path) -> None:
        """Use this method to set the output directory.

        .. warning::
            calling this function will also create the output directory,
            if it does not already exists!

        Args:
        ----
            path (path | str): path to the output path

        """
        self.get_inifile().set_output_directory(path)

    def get_output_directory(self) -> Path:
        """Return the current output directory."""
        return self.get_inifile().get_output_directory()

    def print_configuration(self) -> str:
        """Use this method to obtain string representation of the configuration.

        Use this string to write to file, e.g.:

            >> with open('EmptyProject.ini', 'w') as f:
            >>     f.write(project.print_configuration())

        Returns:
        -------
            (str): string representation of the configuration

        """
        return self.get_inifile().print_configuration()

    @property
    def output_files(self) -> Generator[Path, None, None]:
        """Get a generator object with the output files.

        Yields:
        ------
            Generator[Path, None, None]: generator of output files.

        """
        for of in self._output_files:
            yield self.get_output_directory().joinpath(of)
