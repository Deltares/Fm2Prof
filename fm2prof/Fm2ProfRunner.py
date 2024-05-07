import datetime
import os
import sys
import time
import pickle
import traceback
from pathlib import Path
from typing import Dict, Generator, List, Mapping, NoReturn, Union

import geojson
import numpy as np
import tqdm
from geojson import Feature, FeatureCollection, Polygon
from netCDF4 import Dataset
from scipy.spatial import ConvexHull

from fm2prof import Functions as FE
from fm2prof import __version__
from fm2prof.common import FM2ProfBase
from fm2prof.CrossSection import CrossSection, CrossSectionHelpers
from fm2prof.Export import Export1DModelData, OutputFiles
from fm2prof.Import import FMDataImporter, FmModelData, ImportInputFiles
from fm2prof.IniFile import IniFile
from fm2prof.MaskOutputFile import MaskOutputFile
from fm2prof.RegionPolygonFile import RegionPolygonFile, SectionPolygonFile

class InitializationError(Exception):
    pass

class Fm2ProfRunner(FM2ProfBase):
    __map_key = "2DMapOutput"
    __css_key = "CrossSectionLocationFile"
    __key_frictionweighingmethod = "FrictionweighingMethod"
    __key_skipmaps = "SkipMaps"

    """
    Main class that executes all functionality.

    Arguments:
        iniFilePath (str): path to configuration file
    """

    def __init__(self, iniFilePath: Path | str = ""):
        """
        Initializes the project

        Parameters:
            iniFilePath: path to a configuration file. If not given,
                                default values will be used.
        """
        self.fm_model_data: FmModelData = None
        self._output_files: OutputFiles = OutputFiles()

        self._create_logger()

        iniFilePath = Path(iniFilePath)

        self.start_new_log_task("Loading configuration file")
        try:
            self.load_inifile(iniFilePath)
        except (FileNotFoundError, IOError) as e:
            self.set_logger_message(f"Exiting {e}", "error")
            return

        if not self.get_inifile().has_output_directory:
            self.set_logger_message(
                "Output directory must be set in configuration file", "error"
            )
            return

        # Add a log file
        self.set_logfile(
            output_dir=self.get_inifile().get_output_directory(), filename="fm2prof.log"
        )

        self.finish_log_task()
        # print header to log
        self._print_header()

        # Print configuration to log
        self.set_logger_message(self.get_inifile().print_configuration(), header=True)

    def run(self, overwrite: bool = False) -> None:
        """
        Executes FM2PROF routines.

        Parameters:
            overwrite: if True, overwrites existing output. If False, exits if output detected
        """

        if self.get_inifile() is None:
            self.set_logger_message(
                "No ini file was specified: the run cannot go further.", "Warning"
            )
            return

        # Check for already existing output
        if self._output_exists() & ~overwrite:
            self.set_logger_message(
                "Output already exists. Use overwrite option if you want to re-run the program",
                "warning",
            )
            return

        # Run
        succes = self._run_inifile()

        if not succes:
            self.set_logger_message("Program finished with errors", "warning")
        else:
            self.set_logger_message("Program finished", "info")

    def load_inifile(self, iniFilePath: str):
        """
        use this method to load a configuration file from path.

        Parameters:
            iniFilePath (str): path to configuration file
        """
        IniFileObject = IniFile(iniFilePath, logger=self.get_logger())
        self.set_inifile(IniFileObject)

    def _print_header(self):
        header_text = [
            "=" * 80,
            f"FM2PROF version {__version__}",
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
        """
        Executes main program from the configuration file. 

        The main steps in the program are:
        
        1. Initialize fm2prof
        2. Generate cross-sections
        3. Finalization
        
        Arguments:
            iniFile {IniFile}
                -- Object containing all the information
                    needed to execute the program
        """

        # Initialise the project
        self.start_new_log_task("Initialising FM2PROF")
        try:
            self._initialise_fm2prof()
        except InitializationError:
            return False
        except:
            self.set_logger_message(
                "Unexpected exception during initialisation", "error"
            )
            for line in traceback.format_exc().splitlines():
                self.set_logger_message(line, "debug")
            return False
        self.finish_log_task()

        # Generate cross-sections
        try:
            cross_sections = self._generate_cross_section_list()
        except:
            self.set_logger_message(
                "Unexpected exception during generation of cross-sections. No output produced",
                "error",
            )
            for line in traceback.format_exc().splitlines():
                self.set_logger_message(line, "debug")
            return False

        # Finalise and write output
        self.start_new_log_task("Finalizing")
        try:
            self._finalise_fm2prof(cross_sections)
        except:
            self.set_logger_message("Unexpected exception during finalisation", "error")
            for line in traceback.format_exc().splitlines():
                self.set_logger_message(line, "debug")
            return False

        # Print final report
        try:
            self._print_log_report()
        except:
            self.set_logger_message(
                "Unexpected exception during printing of log report", "error"
            )
        self.finish_log_task()

        return True

    def _initialise_fm2prof(self) -> None:
        """
        
        Loads data, inifile
        """

        iniFile: IniFile = self.get_inifile()
        raiseFileNotFoundError: bool = False

        # shorter local variables
        map_file = iniFile.get_input_file(self.__map_key)
        css_file = iniFile.get_input_file(self.__css_key)
        region_file = iniFile.get_input_file("RegionPolygonFile")
        section_file = iniFile.get_input_file("SectionPolygonFile")
        output_dir = iniFile.get_output_directory()

        # Read region & section polygon
        regions: RegionPolygonFile = None
        sections: SectionPolygonFile = None

        if region_file:
            regions = RegionPolygonFile(region_file, logger=self.get_logger())

        if bool(section_file):
            sections = SectionPolygonFile(section_file, logger=self.get_logger())

        # Check if mandatory input exists
        if not Path(map_file).is_file():
            self.set_logger_message(
                f"File for {self.__map_key} not found at {map_file}", "error"
            )
            raiseFileNotFoundError = True
        if not Path(css_file).is_file():
            self.set_logger_message(
                f"File for {self.__css_key} not found at {css_file}", "error"
            )
            raiseFileNotFoundError = True
        if raiseFileNotFoundError:
            raise InitializationError

        # Read FM model data
        fm2prof_fm_model_data = self._set_fm_model_data(
            map_file, css_file, regions, sections
        )
        self.fm_model_data = FmModelData(fm2prof_fm_model_data)

        # Validate config file
        success:bool = self._validate_config_after_initalization()
        if not success:
            self.set_logger_message("Validation of configuration file not successful. Check the log to fix errors.", "error")
            raise InitializationError
        
        # print goodbye
        ntsteps: int = self.fm_model_data.time_dependent_data.get("waterlevel").shape[1]
        nfaces: int = self.fm_model_data.time_dependent_data.get("waterlevel").shape[0]
        nedges: int = self.fm_model_data.edge_data.get("x").shape[0]
        self.set_logger_message("finished reading FM and cross-sectional data data")
        self.set_logger_message(
            "Number of: timesteps ({}), ".format(ntsteps)
            + "faces ({}), ".format(nfaces)
            + "edges ({})".format(nedges),
            level="debug",
        )
        
        return success

    def _validate_config_after_initalization(self) -> bool:
        """
        Performs validation checks on config file. Returns 
        True if all checks succesfull, False if check fails. 
        """
        success:bool = True

        self.set_logger_message("Validating settings", "Info")

        # Check if skipmaps is lower than maximum amount of maps
        nsteps:int = self.fm_model_data.time_dependent_data.get("waterlevel").shape[1]
        skipmap:int = self.get_inifile().get_parameter(self.__key_skipmaps)

        if skipmap >= nsteps:
            self.set_logger_message(f"""You are attempting to skip more than  available timesteps.
({self.__key_skipmaps} = {skipmap}, available maps in output file: {nsteps}). Modify the value of {self.__key_skipmaps}
your configuration file to fix this error.""", level="error")
            success = False
        elif skipmap > nsteps/2:
            self.set_logger_message(f"""You are skipping more than half of available timesteps.
({self.__key_skipmaps} = {skipmap}, available maps in output file: {nsteps})""", level="warning")
            

        # Check if edge/face data is available
        if "edge_faces" not in self.fm_model_data.edge_data:
            if self.get_inifile().get_parameter(self.__key_frictionweighingmethod) == 1:
                self.set_logger_message(
                    "Friction weighing set to 1 (area-weighted average"
                    + "but FM map file does contain the *edge_faces* keyword."
                    + "Area weighting is not possible. Defaulting to simple unweighted"
                    + "averaging",
                    level="warning",
                )

        return success
    
    def _finalise_fm2prof(self, cross_sections: List) -> None:
        """
        Write to output, perform checks
        """
        self.set_logger_message("Interpolating roughness")
        CrossSectionHelpers().interpolate_friction_across_cross_sections(cross_sections)

        # Export cross sections
        output_dir = self.get_inifile().get_output_directory()
        self.set_logger_message("Export model input files to {}".format(output_dir))
        self._write_output(cross_sections, output_dir)

        # Generate output geojson
        try:
            export_mapfiles = self.get_inifile().get_parameter("ExportMapFiles")
        except KeyError:
            # If key is missing, do not export files by default.
            # We need a better solution for this (inifile.getparam?.. handle defaults there?)
            export_mapfiles = False
        if export_mapfiles:
            self.set_logger_message("Export geojson output to {}".format(output_dir))
            self._generate_geojson_output(output_dir, cross_sections)

        # Export bounding boxes of cross-section control volumes
        try:
            self._export_envelope(output_dir, cross_sections)
        except Exception as e_error:
            e_message = str(e_error)
            self.set_logger_message("Error while exporting bounding boxes", "error")
            self.set_logger_message(e_message, "error")

    def _export_envelope(self, output_dir, cross_sections):
        """
        # Export envelopes around cross-sections
        """
        output = {"type": "FeatureCollection"}
        css_hulls = list()
        for css in cross_sections:
            pointlist = np.array(
                [
                    point["geometry"]["coordinates"]
                    for point in css.get_point_list("face")
                ]
            )
            # construct envelope
            try:
                hull = ConvexHull(pointlist)
                css_hulls.append(
                    Feature(
                        properties={"name": css.name},
                        geometry=Polygon([list(map(tuple, pointlist[hull.vertices]))]),
                    )
                )
            except IndexError:
                self.set_logger_message(f"No Hull Exported For {css.name}")

        with open(os.path.join(output_dir, "cross_section_volumes.geojson"), "w") as f:
            geojson.dump(FeatureCollection(css_hulls), f, indent=2)

    def _set_fm_model_data(self, res_file, css_file, regions, sections):
        """
        Reads input files for 'FM2PROF'. See documentation for file format descriptions.

        Data is saved in three major structures:
            time_independent_data: holds bathymetry information
            time_dependent_data: waterlevels, roughnesses and velocities
            edge_data: the nodes that relate to edges

        :param res_file: str, path to FlowFM map netcfd file (*_map.nc)
        :param css_file: str, path to cross-section definition file
        :param region_file: str, path to region geojson file
        :return:
        """
        importer = ImportInputFiles(logger=self.get_logger())
        ini_file = self.get_inifile()

        # Read FM map file
        self.set_logger_message("Opening FM Map file")
        (
            time_independent_data,
            edge_data,
            node_coordinates,
            time_dependent_data,
        ) = FMDataImporter().import_dflow2d(res_file)
        # (ctime_independent_data, cedge_data, cnode_coordinates, ctime_dependent_data) = FE._read_fm_model(res_file)
        self.set_logger_message("Closed FM Map file")

        # Load locations and names of cross-sections
        self.set_logger_message("Opening css file")
        cssdata = importer.css_file(css_file)
        self.set_logger_message("Closed css file")

        # Classify regions & set cross-sections
        if (ini_file.get_parameter("classificationmethod") == 0) or (regions is None):
            self.set_logger_message(
                "All 2D points assigned to the same region and classifying points to cross-sections"
            )
            time_independent_data, edge_data = FE.classify_without_regions(
                cssdata, time_independent_data, edge_data
            )
        elif ini_file.get_parameter("classificationmethod") == 1:
            self.set_logger_message(
                "Assigning 2D points to regions using DeltaShell and classifying points to cross-sections"
            )
            time_independent_data, edge_data = self._classify_with_deltashell(
                time_independent_data, edge_data, cssdata, regions, polytype="region"
            )
        else:
            self.set_logger_message(
                "Assigning 2D points to regions using Built-In method and classifying points to cross-sections"
            )
            time_independent_data, edge_data = self._classify_with_builtin_methods(
                time_independent_data, edge_data, cssdata, regions
            )

        # Classify sections for roughness tables
        if (ini_file.get_parameter("classificationmethod") == 0) or (sections is None):
            self.set_logger_message("Assigning point to sections without polygons")
            edge_data = self._classify_roughness_sections_by_variance(
                edge_data, time_dependent_data["chezy_edge"]
            )
            time_independent_data = self._classify_roughness_sections_by_variance(
                time_independent_data, time_dependent_data["chezy_mean"]
            )
        elif ini_file.get_parameter("classificationmethod") == 1:
            self.set_logger_message("Assigning 2D points to sections using DeltaShell")
            time_independent_data, edge_data = self._classify_section_with_deltashell(
                time_independent_data, edge_data
            )
        else:
            self.set_logger_message(
                "Assigning 2D points to sections using Built-In method"
            )
            edge_data = FE.classify_roughness_sections_by_polygon(
                sections, edge_data, self.get_logger()
            )
            time_independent_data = FE.classify_roughness_sections_by_polygon(
                sections, time_independent_data, self.get_logger()
            )

        return (
            time_dependent_data,
            time_independent_data,
            edge_data,
            node_coordinates,
            cssdata,
        )

    def _classify_with_builtin_methods(
        self, time_independent_data, edge_data, cssdata, polygons
    ):
        # Determine in which region each cross-section lies
        css_regions = polygons.classify_points(cssdata["xy"])

        # Determine in which region each 2d point lies
        xy_tuples_2d = [
            (
                time_independent_data.get("x").values[i],
                time_independent_data.get("y").values[i],
            )
            for i in range(len(time_independent_data.get("x")))
        ]

        time_independent_data["region"] = regions.classify_points(xy_tuples_2d)

        xy_tuples_2d = [
            (edge_data.get("x")[i], edge_data.get("y")[i])
            for i in range(len(edge_data.get("x")))
        ]

        edge_data["region"] = polygons.classify_points(xy_tuples_2d)

        # Do Nearest neighbour cross-section for each region
        time_independent_data, edge_data = FE.classify_with_regions(
            regions, cssdata, time_independent_data, edge_data, css_regions
        )

        return time_independent_data, edge_data

    def _classify_section_with_deltashell(self, time_independent_data, edge_data):

        # Determine in which section each 2D point lies
        self.set_logger_message("Assigning faces...")
        time_independent_data = self._assign_polygon_using_deltashell(
            time_independent_data, dtype="face", polytype="section"
        )
        self.set_logger_message("Assigning edges...")
        edge_data = self._assign_polygon_using_deltashell(
            edge_data, dtype="edge", polytype="section"
        )

        return time_independent_data, edge_data

    def _classify_with_deltashell(
        self, time_independent_data, edge_data, cssdata, polygons, polytype="region"
    ):

        # Determine in which region each 2D point lies
        self.set_logger_message("Assigning faces...")
        time_independent_data = self._assign_polygon_using_deltashell(
            time_independent_data, dtype="face", polytype=polytype
        )
        self.set_logger_message("Assigning edges...")
        edge_data = self._assign_polygon_using_deltashell(
            edge_data, dtype="edge", polytype=polytype
        )

        self.set_logger_message(
            "Assigning cross-sections using nearest neighbour within regions..."
        )
        # Determine in which region each cross-section lies
        css_regions = polygons.classify_points(cssdata["xy"])

        # Do Nearest neighbour cross-section for each region
        time_independent_data, edge_data = FE.classify_with_regions(
            polygons, cssdata, time_independent_data, edge_data, css_regions
        )

        return time_independent_data, edge_data

    def _classify_roughness_sections_by_variance(self, data, variable):
        """
        This method classifies the region into main channel and floodplain based on roughness. It
        is used when the user does not specify a section polygon.

        This method assumes that the main channel is much deeper than the floodplain. Therefore,
        the Chézy values will be higher than those in the floodplain. The objective is now to
        define a split-value that minimizes the variance of the split sets.

        .. note::
            Variance reduction classification is method often used in decision tree learning,
            e.g. see https://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction
            for more information.

        """
        # Get chezy values at last timestep
        end_values = variable.T.iloc[-1].values
        key = "section"

        # Find split point (chezy value) by variance minimisation
        variance_list = list()
        split_candidates = np.arange(min(end_values), max(end_values), 1)
        if (
            len(split_candidates) < 2
        ):  # this means that all end values are very close together, so do not split
            data[key][:] = 1
        else:
            for split in split_candidates:
                variance_list.append(
                    np.max(
                        [
                            np.var(end_values[end_values > split]),
                            np.var(end_values[end_values <= split]),
                        ]
                    )
                )

            splitpoint = split_candidates[np.nanargmin(variance_list)]

            # High chezy values are assigned to section number '1' (Main channel)
            data[key][end_values > splitpoint] = 1

            # Low chezy values are assigned to section number '2' (Flood plain)
            data[key][end_values <= splitpoint] = 2
        return data

    def _get_region_map_file(self, polytype):
        """Returns the path to a NC file with region ifnormation in the bathymetry data"""
        map_file_path = self.get_inifile().get_input_file("2DMapOutput")
        filepath, ext = os.path.splitext(map_file_path)
        modified_file_path = f"{filepath}_{polytype.upper()}BATHY{ext}"
        return modified_file_path

    def _assign_polygon_using_deltashell(
        self, data, dtype: str = "face", polytype: str = "region"
    ):
        """Assign all 2D points using DeltaShell method"""

        # NOTE
        self.set_logger_message(f"Looking for _{polytype.upper()}BATHY.nc", "debug")

        path_to_modified_nc = self._get_region_map_file(polytype)

        # Load Modified NetCDF
        with Dataset(path_to_modified_nc) as nf:
            # Data stored in node z, while fm2prof uses data at faces or edges.
            region_at_node = nf.variables.get("mesh2d_node_z")[:].data.astype(int)

            if dtype == "face":
                node_to_face = nf.variables.get("mesh2d_face_nodes")[:].data
                region_at_face = region_at_node[node_to_face.T[0] - 1]
                data[polytype] = region_at_face
            elif dtype == "edge":
                node_to_edge = data["edge_nodes"]
                region_at_edge = region_at_node[node_to_edge.T[0] - 1]
                data[polytype] = region_at_edge

        return data

    def _generate_geojson_output(self, output_dir: str, cross_sections: list):
        """Generates geojson file based on cross sections.

        Arguments:
            output_dir {str} -- Output directory path.
            cross_sections {list} -- List of Cross Sections.
        """
        for pointtype in ["face", "edge"]:
            output_file_path = os.path.join(
                output_dir, "{}_output.geojson".format(pointtype)
            )
            try:
                node_points = [
                    node_point
                    for cs in cross_sections
                    for node_point in cs.get_point_list(pointtype)
                ]
                self.set_logger_message(
                    "Collected points, dumping to file", level="debug"
                )
                MaskOutputFile.write_mask_output_file(output_file_path, node_points)
                self.set_logger_message("Done", level="debug")
            except Exception as e_info:
                self.set_logger_message(
                    "Error while generation .geojson file,"
                    + "at {}".format(output_file_path)
                    + "Reason: {}".format(str(e_info)),
                    level="error",
                )

    def _generate_cross_section_list(self):
        """Generates cross sections based on the given fm_model_data

        Arguments:
            input_param_dict {Mapping[str, list]}
                -- Dictionary of parameters read from IniFile
            fm_model_data {FmModelData}
                -- Class with all necessary data for generating Cross Sections

        Returns:
            {list} -- List of generated cross sections
        """

        cross_sections = list()
        if not self.fm_model_data:
            return cross_sections

        # Preprocess css from fm_model_data so it's easier to handle it.
        css_data_list = self.fm_model_data.css_data_list

        # Set the number of cross-section for progress bar
        css_selection = self._get_css_range(number_of_css=len(css_data_list))
        self.get_logformatter().set_number_of_iterations(len(css_selection) + 1)
        selected_list = np.array(css_data_list)[css_selection]

        # Generate cross-sections one by one
        pbar = tqdm.tqdm(total=len(selected_list))
        for i, css_data in enumerate(selected_list):
            self.start_new_log_task(
                f"{css_data.get('id')}  ({i}/{len(selected_list)})", pbar=pbar
            )
            generated_cross_section = self._generate_cross_section(
                css_data, self.fm_model_data
            )
            if generated_cross_section is not None:
                cross_sections.append(generated_cross_section)
            pbar.update(1)

        return cross_sections

    def _get_css_range(self, number_of_css: int):
        """parses the CssSelection keyword from the inifile"""
        cssSelection = self.get_inifile().get_parameter("CssSelection")
        if not cssSelection:
            cssSelection = np.arange(0, number_of_css)
        else:
            cssSelection = np.array(cssSelection)
        return cssSelection

    def _generate_cross_section(
        self, css_data: Dict, fm_model_data: FmModelData
    ) -> CrossSection:
        """Generates a cross section and configures its values based
        on the input parameter dictionary

        Arguments:
            css_data {dict}
                -- Dictionary of data for the current cross section
            input_param_dict {Mapping[str,list]}
                -- Dictionary with input parameters
            fm_model_data {FmModelData}
                -- Data to assign to the new cross section

        Raises:
            Exception: If no css_data is given.
            Exception: If no input_param_dict is given.
            Exception: If no fm_model_data is given.

        Returns:
            {CrossSection} -- New Cross Section
        """
        if css_data is None:
            raise Exception("No data was given to create a Cross Section")

        css_name = css_data.get("id")
        if not css_name:
            css_name = "new_cross_section"

        if fm_model_data is None:
            raise Exception(
                "No FM data given for new cross section {}".format(css_name)
            )
        
        # Create cross section
        created_css = self._create_new_cross_section(css_data=css_data)

        if created_css is None:
            self.set_logger_message(
                f"No Cross-section could be generated for {css_name}", "error"
            )
            return None
        if created_css.get_number_of_faces() < 10:
            self.set_logger_message(
                f"There are too little 2D points in control volume to construct cross-section",
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
        self, cross_section: CrossSection
    ) -> CrossSection:
        """
        This method manages the options of building the cross-section geometry

        Parameters:
            cross_section {CrossSection}
                -- Given Cross Section.
        """

        if cross_section is None:
            raise Exception

        # Build cross-section
        self.set_logger_message("Start building geometry", "debug")
        cross_section.build_geometry()

        # 2D Volume Correction (SummerDike option)
        if self.get_inifile().get_parameter("SDCorrection"):
            self.set_logger_message("Starting correction", "debug")
            cross_section = self._perform_2D_volume_correction(cross_section)
        else:
            self.set_logger_message(
                "SD Correction not enable in configuration file, skipping", "info"
            )

        # Perform sanity check on cross-section
        cross_section.check_requirements()

        # Reduce number of points in cross-section
        cross_section = self._reduce_css_points(cross_section)

        return cross_section

    def _build_cross_section_roughness(
        self, cross_section: CrossSection
    ) -> CrossSection:
        """
        Build the roughness tables
        """
        # Assign roughness
        self.set_logger_message("Starting computing roughness tables", "debug")
        cross_section.assign_roughness()
        self.set_logger_message("Computed roughness", "info")

        return cross_section

    def _create_new_cross_section(self, css_data: Mapping[str, str]):
        """Creates a cross section with the given input param dictionary.

        Arguments:
            css_data {Mapping[str, str]}
                -- FM Model data for cross section.
            input_param_dict {Mapping[str, str]}
                -- Dictionary with parameters for Cross Section.

        Returns:
            {CrossSection} -- New cross section object.
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
        css_data['fm_data'] = self.fm_model_data.get_selection(css_data.get("id"))

        if self.get_inifile().get_parameter("ExportCSSData"):
            output_dir = self.get_inifile().get_output_directory()
            with open(output_dir.joinpath(f"{css_data.get('id')}.pickle"), 'wb') as f:
                pickle.dump(css_data, f)
        try:
            css = CrossSection(
                logger=self.get_logger(),
                inifile=self.get_inifile(),
                data = css_data,
            )

        except Exception as e_info:
            self.set_logger_message(
                "Exception thrown while creating cross-section "
                + "{}, message: {}".format(css_data_id, str(e_info)),
                "error",
            )
            return None

        return css

    def _write_output(self, cross_sections: list, output_dir: Path):
        """Exports all cross sections to the necessary file formats

        Arguments:
            cross_sections {list}
                -- List of created cross sections
            output_dir {str}
                -- target directory where to export all the cross sections
        """
        if not cross_sections:
            return

        if not output_dir or not os.path.exists(output_dir):
            return

        OutputExporter = Export1DModelData(logger=self.get_logger())

        # File paths
        css_location_ini_file = output_dir.joinpath(
            self._output_files.dimr_css_locations
        )
        css_definitions_ini_file = output_dir.joinpath(
            self._output_files.dimr_css_definitions
        )

        # Legacy file formats
        csv_geometry_file = output_dir.joinpath(self._output_files.sobek3_geometry)
        csv_roughness_file = output_dir.joinpath(self._output_files.sobek3_roughness)

        csv_geometry_test_file = output_dir.joinpath(self._output_files.test_geometry)
        csv_roughness_test_file = output_dir.joinpath(self._output_files.test_roughness)

        csv_volumes_file = output_dir.joinpath(self._output_files.fm2prof_volume)

        # export fm1D format
        try:
            # Export locations
            OutputExporter.export_crossSectionLocations(
                cross_sections, file_path=css_location_ini_file
            )

            # Export definitions
            OutputExporter.export_geometry(
                cross_sections, file_path=css_definitions_ini_file, fmt="dflow1d"
            )

            # Export roughness
            sections = np.unique(
                [s for css in cross_sections for s in css.friction_tables.keys()]
            )
            sectionFileKeyDict = {
                "main": [self._output_files.dimr_roughness_main, "Main"],
                "floodplain1": [
                    self._output_files.dimr_roughness_floodplain1,
                    "FloodPlain1",
                ],
                "floodplain2": [
                    self._output_files.dimr_roughness_floodplain2,
                    "FloodPlain2",
                ],
            }
            for section in sections:
                csv_roughness_ini_file = output_dir.joinpath(
                    sectionFileKeyDict[section][0]
                )
                OutputExporter.export_roughness(
                    cross_sections,
                    file_path=csv_roughness_ini_file,
                    fmt="dflow1d",
                    roughness_section=sectionFileKeyDict[section][1],
                )

        except Exception as e_info:
            self.set_logger_message(
                "An error was produced while exporting files to DIMR format,"
                + " not all output files might be exported. "
                + "{}".format(str(e_info)),
                level="error",
            )

        # Eport SOBEK 3 format
        try:
            # Cross-sections
            OutputExporter.export_geometry(
                cross_sections, file_path=csv_geometry_file, fmt="sobek3"
            )

            # Roughness
            OutputExporter.export_roughness(
                cross_sections, file_path=csv_roughness_file, fmt="sobek3"
            )
        except Exception as e_info:
            self.set_logger_message(
                "An error was produced while exporting files to SOBEK format,"
                + " not all output files might be exported. "
                + "{}".format(str(e_info)),
                level="error",
            )

        # Other files:
        try:
            OutputExporter.export_geometry(
                cross_sections, file_path=csv_geometry_test_file, fmt="testformat"
            )

            OutputExporter.export_volumes(cross_sections, file_path=csv_volumes_file)
        except Exception as e_info:
            self.set_logger_message(
                "An error was produced while exporting files,"
                + " not all output files might be exported. "
                + "{}".format(str(e_info)),
                level="error",
            )

        self.set_logger_message("Exported output files, FM2PROF finished")

    def _reduce_css_points(self, cross_section: CrossSection):
        """Returns a valid value for the number of css points read from ini file.

        Parameters:
            cross_section (CrossSection)

        Returns:
            cross_section (CrossSection): modified
        """

        maximum_number_of_css_points = self.get_inifile().get_parameter(
            "MaximumPointsInProfile"
        )

        try:
            cross_section.reduce_points(count_after=maximum_number_of_css_points)
        except Exception as e_error:
            e_message = str(e_error)
            self.set_logger_message(
                "Exception thrown while trying to reduce the css points. "
                + "{}".format(e_message),
                "error",
            )

        return cross_section

    def _get_time_stamp_seconds(self, start_time: datetime):
        """Returns a time stamp with the time difference

        Arguments:
            start_time {datetime} -- Initial date time

        Returns:
            {float} -- difference of time between start and now in seconds
        """
        time_now = datetime.datetime.now()
        time_difference = time_now - start_time
        return time_difference.total_seconds()

    def _perform_2D_volume_correction(self, css: CrossSection) -> CrossSection:
        """
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
                "Exception thrown "
                + "while trying to calculate the correction. "
                + "{}".format(e_message),
                "error",
            )
        return css

    def _print_log_report(self) -> None:
        ll = self.get_logformatter()._loglibrary
        self.set_logger_message(f"Warnings: {ll.get('WARNING')}")
        self.set_logger_message(f"Errors: {ll.get('ERROR')}")

    def _output_exists(self) -> bool:
        """
        Checks whether output exists
        """
        for output_file in self._output_files:
            if (
                self.get_inifile()
                .get_output_directory()
                .joinpath(output_file)
                .is_file()
            ):
                return True
        else:
            return False


class Project(Fm2ProfRunner):
    """
    Provides the python API for running FM2PROF.

    Instantiate by providing the path to a configuration file

    >> Project('/path/to/config.ini')

    """

    def set_parameter(self, name: str, value: Union[str, float, int]):
        """
        Use this method to set the value of a parameter

        Arguments:
            name: name of the parameter (case insensitive).

            value: value of the parameter. An error will be given if the value has the wrong type (e.g. string if int was expected).
        """
        self.get_inifile().set_parameter(name, value)

    def get_parameter(self, name: str) -> Union[str, float, int]:
        """
        Use this method to get the value of a parameter

        Arguments:
            name: name of the parameter (case insensitive)

        Returns:
            The current value of the parameter
        """
        return self.get_inifile().get_parameter(name)

    def set_input_file(self, name: str, value: Union[str, float, int]) -> None:
        """
        Use this method to set the path to an input file

        Arguments:
            name: name of the input file in the configuration (case insensitive).

            value: path to the inputfile
        """
        return self.get_inifile().set_input_file(name, value)

    def get_input_file(
        self, name: str
    ) -> Union[str,]:
        """
        Use this method to retrieve the path to an input file

        Arguments:
            name (str): case-insensitive key of the input file (e.g.'2dmapoutput')
        """
        return self.get_inifile().get_input_file(name)

    def set_output_directory(self, path) -> None:
        """
        Use this method to set the output directory.

        .. warning::
            calling this function will also create the output directory,
            if it does not already exists!

        Arguments:
            path: path to the output path
        """
        self.get_inifile().set_output_directory(path)

    def get_output_directory(self) -> str:
        """
        Returns the current output directory
        """
        return self.get_inifile().get_output_directory()

    def print_configuration(self) -> str:
        """
        Use this method to obtain string representation of the
        configuration. Use this string to write to file, e.g.:

            >> with open('EmptyProject.ini', 'w') as f:
            >>     f.write(project.print_configuration())

        Returns:
            string

        """
        return self.get_inifile().print_configuration()

    @property
    def output_files(self) -> Generator[Path, None, None]:
        for of in self._output_files:
            yield self.get_output_directory().joinpath(of)
