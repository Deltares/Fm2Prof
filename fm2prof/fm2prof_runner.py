"""Module for running Fm2Prof processess."""

from __future__ import annotations

import datetime
import pickle
import traceback
from pathlib import Path
from typing import Generator

import geojson
import numpy as np
import pandas as pd
import tqdm
from geojson import Feature, FeatureCollection, Polygon
from netCDF4 import Dataset
from scipy.spatial import ConvexHull
from sklearn.neighbors import KNeighborsClassifier

from fm2prof import __version__, mask_output_file
from fm2prof.common import FM2ProfBase
from fm2prof.cross_section import CrossSection, CrossSectionHelpers
from fm2prof.data_import import FMDataImporter, FmModelData, ImportInputFiles
from fm2prof.export import Export1DModelData, OutputFiles
from fm2prof.ini_file import IniFile
from fm2prof.region_polygon_file import RegionPolygonFile, SectionPolygonFile


class InitializationError(Exception):
    """Exception class for initialization errors."""


class Fm2ProfRunner(FM2ProfBase):
    """Main class that executes all functionality."""

    __map_key = "2DMapOutput"
    __css_key = "CrossSectionLocationFile"
    __key_frictionweighingmethod = "FrictionweighingMethod"
    __key_skipmaps = "SkipMaps"

    def __init__(self, ini_file_path: Path | str = "") -> None:
        """Initialize the project.

        Args:
        ----
            ini_file_path (Path | str): path to configuration file.

        """
        self.fm_model_data: FmModelData = None
        self._output_files: OutputFiles = OutputFiles()

        self._create_logger()

        ini_file_path = Path(ini_file_path)

        self.start_new_log_task("Loading configuration file")
        try:
            self.load_inifile(ini_file_path)
        except (OSError, FileNotFoundError) as e:
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

    def run(self, *, overwrite: bool = False) -> None:
        """Execute FM2PROF routines.

        Args:
        ----
            overwrite (bool): if True, overwrites existing output. If False, exits if output detected.

        """
        if self.get_inifile() is None:
            self.set_logger_message(
                "No ini file was specified: the run cannot go further.",
                "Warning",
            )
            return

        # Check for already existing output
        if self._output_exists() and not overwrite:
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

    def load_inifile(self, ini_file_path: str) -> None:
        """Use this method to load a configuration file from path.

        Args:
        ----
            ini_file_path (str): path to configuration file

        """
        ini_file_object = IniFile(ini_file_path, logger=self.get_logger())
        self.set_inifile(ini_file_object)

    def _print_header(self) -> None:
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
        """Execute main program from the configuration file.

        The main steps in the program are:

        1. Initialize fm2prof
        2. Generate cross-sections
        3. Finalization

        """
        # Initialise the project
        self.start_new_log_task("Initialising FM2PROF")
        try:
            self._initialise_fm2prof()
        except InitializationError:
            return False
        except:
            self.set_logger_message(
                "Unexpected exception during initialisation",
                "error",
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
                "Unexpected exception during printing of log report",
                "error",
            )
        self.finish_log_task()

        return True

    def _initialise_fm2prof(self) -> None:
        """Load data, inifile."""
        ini_file: IniFile = self.get_inifile()
        raise_file_not_found: bool = False

        # shorter local variables
        map_file = ini_file.get_input_file(self.__map_key)
        css_file = ini_file.get_input_file(self.__css_key)
        region_file = ini_file.get_input_file("RegionPolygonFile")
        section_file = ini_file.get_input_file("SectionPolygonFile")

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
                f"File for {self.__map_key} not found at {map_file}",
                "error",
            )
            raise_file_not_found = True
        if not Path(css_file).is_file():
            self.set_logger_message(
                f"File for {self.__css_key} not found at {css_file}",
                "error",
            )
            raise_file_not_found = True
        if raise_file_not_found:
            raise InitializationError

        # Read FM model data
        (
            time_dependent_data,
            time_independent_data,
            edge_data,
            node_coordinates,
            css_data_dictionary,
        ) = self._set_fm_model_data(
            map_file,
            css_file,
            regions,
            sections,
        )
        self.fm_model_data = FmModelData(
            time_dependent_data=time_dependent_data,
            time_independent_data=time_independent_data,
            edge_data=edge_data,
            node_coordinates=node_coordinates,
            css_data_dictionary=css_data_dictionary,
        )

        # Validate config file
        success: bool = self._validate_config_after_initalization()
        if not success:
            self.set_logger_message(
                "Validation of configuration file not successful. Check the log to fix errors.",
                "error",
            )
            raise InitializationError

        # print goodbye
        ntsteps: int = self.fm_model_data.time_dependent_data.get("waterlevel").shape[1]
        nfaces: int = self.fm_model_data.time_dependent_data.get("waterlevel").shape[0]
        nedges: int = self.fm_model_data.edge_data.get("x").shape[0]
        self.set_logger_message("finished reading FM and cross-sectional data data")
        self.set_logger_message(
            f"Number of: timesteps ({ntsteps}), " + f"faces ({nfaces}), " + f"edges ({nedges})",
            level="debug",
        )

        return success

    def _validate_config_after_initalization(self) -> bool:
        """Perform validation checks on config file.

        Returns True if all checks succesfull, False if check fails.
        """
        success: bool = True

        self.set_logger_message("Validating settings", "Info")

        # Check if skipmaps is lower than maximum amount of maps
        nsteps: int = self.fm_model_data.time_dependent_data.get("waterlevel").shape[1]
        skipmap: int = self.get_inifile().get_parameter(self.__key_skipmaps)

        if skipmap >= nsteps:
            self.set_logger_message(
                f"""You are attempting to skip more than  available timesteps.
({self.__key_skipmaps} = {skipmap}, available maps in output file: {nsteps}). Modify the value of {self.__key_skipmaps}
your configuration file to fix this error.""",
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
            "edge_faces" not in self.fm_model_data.edge_data
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

    def _finalise_fm2prof(self, cross_sections: list[CrossSection]) -> None:
        """Write to output, perform checks."""
        self.set_logger_message("Interpolating roughness")
        CrossSectionHelpers().interpolate_friction_across_cross_sections(cross_sections)

        # Export cross sections
        output_dir = self.get_inifile().get_output_directory()
        self.set_logger_message(f"Export model input files to {output_dir}")
        self._write_output(cross_sections, output_dir)

        # Generate output geojson
        try:
            export_mapfiles = self.get_inifile().get_parameter("ExportMapFiles")
        except KeyError:
            # If key is missing, do not export files by default.
            # We need a better solution for this (inifile.getparam?.. handle defaults there?)
            export_mapfiles = False
        if export_mapfiles:
            self.set_logger_message(f"Export geojson output to {output_dir}")
            self._generate_geojson_output(output_dir, cross_sections)

        # Export bounding boxes of cross-section control volumes
        try:
            self._export_envelope(output_dir, cross_sections)
        except Exception as e_error:
            e_message = str(e_error)
            self.set_logger_message("Error while exporting bounding boxes", "error")
            self.set_logger_message(e_message, "error")

    def _export_envelope(
        self,
        output_dir: Path | str,
        cross_sections: list[CrossSection],
    ) -> None:
        """Export envelopes around cross-sections."""
        css_hulls = []
        for css in cross_sections:
            pointlist = np.array(
                [point["geometry"]["coordinates"] for point in css.get_point_list("face")],
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

    def _set_fm_model_data(
        self,
        res_file: str | Path,
        css_file: str | Path,
        regions: RegionPolygonFile | None,
        sections: SectionPolygonFile | None,
    ) -> tuple:
        """Read input files for 'FM2PROF'.

        See documentation for file format descriptions.

        Args:
            res_file (str | Path): path to FlowFM map netcfd file (*_map.nc)
            css_file (str | Path): path to cross-section definition file_
            regions (RegionPolygonFile | None): RegionPolygonFile object
            sections (SectionPolygonFile | None): SectionPolygonFile object

        Returns:
            tuple: Tuple containing time dependent data, time independent data, edge data, node coordinates,
            and cross section data.

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
        self.set_logger_message("Closed FM Map file")

        # Load locations and names of cross-sections
        self.set_logger_message("Opening css file")
        cssdata = importer.css_file(css_file)
        self.set_logger_message("Closed css file")

        # Classify regions & set cross-sections
        if (ini_file.get_parameter("classificationmethod") == 0) or (regions is None):
            self.set_logger_message(
                "All 2D points assigned to the same region and classifying points to cross-sections",
            )
            time_independent_data, edge_data = self._classify_without_regions(
                cssdata,
                time_independent_data,
                edge_data,
            )
        elif ini_file.get_parameter("classificationmethod") == 1:
            self.set_logger_message(
                "Assigning 2D points to regions using DeltaShell and classifying points to cross-sections",
            )
            time_independent_data, edge_data = self._classify_with_deltashell(
                time_independent_data,
                edge_data,
                cssdata,
                regions,
                polytype="region",
            )
        else:
            self.set_logger_message(
                "Assigning 2D points to regions using Built-In method and classifying points to cross-sections",
            )
            time_independent_data, edge_data = self._classify_with_builtin_methods(
                time_independent_data,
                edge_data,
                cssdata,
                regions,
            )

        # Classify sections for roughness tables
        if (ini_file.get_parameter("classificationmethod") == 0) or (sections is None):
            self.set_logger_message("Assigning point to sections without polygons")
            edge_data = self._classify_roughness_sections_by_variance(
                edge_data,
                time_dependent_data["chezy_edge"],
            )
            time_independent_data = self._classify_roughness_sections_by_variance(
                time_independent_data,
                time_dependent_data["chezy_mean"],
            )
        elif ini_file.get_parameter("classificationmethod") == 1:
            self.set_logger_message("Assigning 2D points to sections using DeltaShell")
            time_independent_data, edge_data = self._classify_section_with_deltashell(
                time_independent_data,
                edge_data,
            )
        else:
            self.set_logger_message(
                "Assigning 2D points to sections using Built-In method",
            )
            edge_data = self._classify_roughness_sections_by_polygon(
                sections,
                edge_data,
                self.get_logger(),
            )
            time_independent_data = self._classify_roughness_sections_by_polygon(
                sections,
                time_independent_data,
                self.get_logger(),
            )

        return (
            time_dependent_data,
            time_independent_data,
            edge_data,
            node_coordinates,
            cssdata,
        )

    def _classify_with_builtin_methods(
        self,
        time_independent_data: pd.DataFrame,
        edge_data: dict,
        cssdata: dict,
        regions: RegionPolygonFile,
    ) -> tuple[pd.DataFrame, dict]:
        # Determine in which region each cross-section lies
        css_regions = regions.classify_points(cssdata["xy"])

        # Determine in which region each 2d point lies

        nr_of_time_independent_data_values = len(time_independent_data.get("x"))
        x_tid_array = time_independent_data.get("x").to_numpy()
        y_tid_array = time_independent_data.get("y").to_numpy()

        xy_tuples_2d = [
            (
                x_tid_array[i],
                y_tid_array[i],
            )
            for i in range(nr_of_time_independent_data_values)
        ]

        time_independent_data["region"] = regions.classify_points(xy_tuples_2d)

        xy_tuples_2d = [(edge_data.get("x")[i], edge_data.get("y")[i]) for i in range(len(edge_data.get("x")))]

        edge_data["region"] = regions.classify_points(xy_tuples_2d)

        # Do Nearest neighbour cross-section for each region
        time_independent_data, edge_data = self._classify_with_regions(
            regions,
            cssdata,
            time_independent_data,
            edge_data,
            css_regions,
        )

        return time_independent_data, edge_data

    def _classify_section_with_deltashell(
        self,
        time_independent_data: pd.DataFrame,
        edge_data: dict,
    ) -> tuple[pd.DataFrame, dict]:
        # Determine in which section each 2D point lies
        self.set_logger_message("Assigning faces...")
        time_independent_data = self._assign_polygon_using_deltashell(
            time_independent_data,
            dtype="face",
            polytype="section",
        )
        self.set_logger_message("Assigning edges...")
        edge_data = self._assign_polygon_using_deltashell(
            edge_data,
            dtype="edge",
            polytype="section",
        )

        return time_independent_data, edge_data

    def _classify_with_deltashell(
        self,
        time_independent_data: pd.DataFrame,
        edge_data: dict,
        cssdata: dict,
        polygons: RegionPolygonFile,
        polytype: str = "region",
    ) -> tuple[pd.DataFrame, dict]:
        # Determine in which region each 2D point lies
        self.set_logger_message("Assigning faces...")
        time_independent_data = self._assign_polygon_using_deltashell(
            time_independent_data,
            dtype="face",
            polytype=polytype,
        )
        self.set_logger_message("Assigning edges...")
        edge_data = self._assign_polygon_using_deltashell(
            edge_data,
            dtype="edge",
            polytype=polytype,
        )

        self.set_logger_message(
            "Assigning cross-sections using nearest neighbour within regions...",
        )
        # Determine in which region each cross-section lies
        css_regions = polygons.classify_points(cssdata["xy"])

        # Do Nearest neighbour cross-section for each region
        time_independent_data, edge_data = self._classify_with_regions(
            polygons,
            cssdata,
            time_independent_data,
            edge_data,
            css_regions,
        )

        return time_independent_data, edge_data

    def _classify_roughness_sections_by_variance(
        self,
        data: pd.DataFrame | dict,
        variable: pd.DataFrame,
    ) -> pd.DataFrame | dict:
        """Classify the region into main channel and floodplain based on roughness.

        It is used when the user does not specify a section polygon.

        This method assumes that the main channel is much deeper than the floodplain. Therefore,
        the Chézy values will be higher than those in the floodplain. The objective is now to
        define a split-value that minimizes the variance of the split sets.

        .. note::
            Variance reduction classification is method often used in decision tree learning,
            e.g. see https://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction
            for more information.

        """
        # Get chezy values at last timestep
        end_values = variable.T.iloc[-1].to_numpy()
        key = "section"

        # Find split point (chezy value) by variance minimisation
        split_candidates = np.arange(min(end_values), max(end_values), 1)
        if len(split_candidates) < 2:  # noqa: PLR2004, this means that all end values are very close together, so do not split
            data[key][:] = 1
        else:
            variance_list = [
                np.max(
                    [
                        np.var(end_values[end_values > split]),
                        np.var(end_values[end_values <= split]),
                    ],
                )
                for split in split_candidates
            ]
            splitpoint = split_candidates[np.nanargmin(variance_list)]

            # High chezy values are assigned to section number '1' (Main channel)
            # Low chezy values are assigned to section number '2' (Flood plain)
            if isinstance(data, pd.DataFrame):
                data.loc[end_values > splitpoint, key] = 1
                data.loc[end_values <= splitpoint, key] = 2
            else:
                data[key][end_values > splitpoint] = 1
                data[key][end_values <= splitpoint] = 2
        return data

    def _classify_with_regions(
        self,
        cssdata: dict,
        time_independent_data: pd.DataFrame,
        edge_data: dict,
        css_regions: list,
    ) -> tuple[pd.DataFrame, dict]:
        """Assign cross-section id's based on region polygons.

        Within a region, assignment will be done by k nearest neighbour
        """
        time_independent_data["sclass"] = time_independent_data["region"].astype(str)
        # Nearest Neighbour within regions
        for region in np.unique(css_regions):
            # Select cross-sections within this region
            css_xy = cssdata["xy"][css_regions == region]
            css_id = cssdata["id"][css_regions == region]

            # Select 2d points within region
            node_mask = time_independent_data["region"] == region
            x_2d_node = time_independent_data["x"][node_mask]
            y_2d_node = time_independent_data["y"][node_mask]

            edge_mask = edge_data["region"] == region
            x_2d_edge = edge_data["x"][edge_mask]
            y_2d_edge = edge_data["y"][edge_mask]

            # Do Nearest Neighour
            neigh = self._get_classification_tree(css_xy, css_id)
            css_2d_nodes = neigh.predict(np.array([x_2d_node, y_2d_node]).T)
            css_2d_edges = neigh.predict(np.array([x_2d_edge, y_2d_edge]).T)

            # Update data in main structures
            time_independent_data.loc[node_mask, "sclass"] = css_2d_nodes  # sclass = cross-section id

            edge_data["sclass"][edge_mask] = css_2d_edges

        return time_independent_data, edge_data

    def _classify_without_regions(
        self,
        cssdata: dict,
        time_independent_data: pd.DataFrame,
        edge_data: dict,
    ) -> tuple[pd.DataFrame, dict]:
        """Classify without regions."""
        # Create a class identifier to map points to cross-sections
        neigh = self._get_classification_tree(cssdata["xy"], cssdata["id"])

        # Expand time-independent dataset with cross-section names
        time_independent_data["sclass"] = neigh.predict(
            np.array([time_independent_data["x"], time_independent_data["y"]]).T,
        )

        # Assign cross-section names to edge coordinates as well
        edge_data["sclass"] = neigh.predict(np.array([edge_data["x"], edge_data["y"]]).T)

        return time_independent_data, edge_data

    def _classify_roughness_sections_by_polygon(
        self,
        sections: SectionPolygonFile,
        data: dict | pd.DataFrame,
    ) -> pd.DataFrame | dict:
        """Assign edges to a roughness section based on polygon data."""
        self.set_logger_message("....gathering points", "debug")
        points = [(data["x"][i], data["y"][i]) for i in range(len(data["x"]))]
        self.set_logger_message("....classifying points", "debug")
        data["section"] = sections.classify_points(points)
        return data

    def _get_classification_tree(self, xy: np.ndarray, c: np.ndarray) -> KNeighborsClassifier:
        x = xy
        y = c
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(x, y)
        return neigh

    def _get_region_map_file(self, polytype: str) -> str:
        """Return the path to a NC file with region ifnormation in the bathymetry data."""
        map_file_path = Path(self.get_inifile().get_input_file("2DMapOutput"))
        return f"{map_file_path.parent / map_file_path.stem}_{polytype.upper()}BATHY{map_file_path.suffix}"

    def _assign_polygon_using_deltashell(
        self,
        data: dict | pd.DataFrame,
        dtype: str = "face",
        polytype: str = "region",
    ) -> pd.DataFrame | dict:
        """Assign all 2D points using DeltaShell method."""
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
                node_points = [node_point for cs in cross_sections for node_point in cs.get_point_list(pointtype)]
                self.set_logger_message(
                    "Collected points, dumping to file",
                    level="debug",
                )
                mask_output_file.write_mask_output_file(output_file_path, node_points)
                self.set_logger_message("Done", level="debug")
            except Exception as e_info:
                self.set_logger_message(
                    "Error while generation .geojson file," f"at {output_file_path}" f"Reason: {e_info!s}",
                    level="error",
                )

    def _generate_cross_section_list(self) -> list[CrossSection]:
        """Generate cross sections based on the given fm_model_data.

        Returns:
        -------
            (list): List of generated cross sections

        """
        cross_sections = []
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
                f"{css_data.get('id')}  ({i}/{len(selected_list)})",
                pbar=pbar,
            )
            generated_cross_section = self._generate_cross_section(
                css_data,
                self.fm_model_data,
            )
            if generated_cross_section is not None:
                cross_sections.append(generated_cross_section)
            pbar.update(1)

        return cross_sections

    def _get_css_range(self, number_of_css: int) -> np.array:
        """Parse the CssSelection keyword from the inifile."""
        css_selection = self.get_inifile().get_parameter("CssSelection")
        return np.arange(0, number_of_css) if not css_selection else np.array(css_selection)

    def _generate_cross_section(
        self,
        css_data: dict,
        fm_model_data: FmModelData,
    ) -> CrossSection:
        """Generate a cross section and configures its values based.

        on the input parameter dictionary

        Args:
        ----
            css_data (dict): Dictionary of data for the current cross section.
            fm_model_data (FmModelData): Data to assign to the new cross section

        Raises:
        ------
            Exception: If no css_data is given.
            Exception: If no input_param_dict is given.
            Exception: If no fm_model_data is given.

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

        if fm_model_data is None:
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
                "There are too little 2D points in control volume to construct cross-section",
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

        # Perform sanity check on cross-section
        cross_section.check_requirements()

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
        css_data["fm_data"] = self.fm_model_data.get_selection(css_data.get("id"))

        if self.get_inifile().get_parameter("ExportCSSData"):
            output_dir = Path(self.get_inifile().get_output_directory())
            with output_dir.joinpath(f"{css_data.get('id')}.pickle").open("wb") as f:
                pickle.dump(css_data, f)
        try:
            css = CrossSection(
                logger=self.get_logger(),
                inifile=self.get_inifile(),
                data=css_data,
            )

        except Exception as e_info:
            self.set_logger_message(
                "Exception thrown while creating cross-section " + f"{css_data.get('id')}, message: {e_info!s}",
                "error",
            )
            return None

        return css

    def _write_output(self, cross_sections: list, output_dir: Path) -> None:
        """Export all cross sections to the necessary file formats.

        Args:
        ----
            cross_sections (list): List of created cross sections
            output_dir (str): target directory where to export all the cross sections

        """
        if not cross_sections or not output_dir.exists():
            return

        output_exporter = Export1DModelData(logger=self.get_logger())

        # File paths
        css_location_ini_file = output_dir.joinpath(
            self._output_files.dimr_css_locations,
        )
        css_definitions_ini_file = output_dir.joinpath(
            self._output_files.dimr_css_definitions,
        )

        # Legacy file formats
        csv_geometry_file = output_dir.joinpath(self._output_files.sobek3_geometry)
        csv_roughness_file = output_dir.joinpath(self._output_files.sobek3_roughness)

        csv_geometry_test_file = output_dir.joinpath(self._output_files.test_geometry)
        csv_volumes_file = output_dir.joinpath(self._output_files.fm2prof_volume)

        # export fm1D format
        try:
            # Export locations
            output_exporter.export_cross_section_locations(
                cross_sections,
                file_path=css_location_ini_file,
            )

            # Export definitions
            output_exporter.export_geometry(
                cross_sections,
                file_path=css_definitions_ini_file,
                fmt="dflow1d",
            )

            # Export roughness
            sections = np.unique(
                [s for css in cross_sections for s in css.friction_tables],
            )
            section_file_key_dict = {
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
                    section_file_key_dict[section][0],
                )
                output_exporter.export_roughness(
                    cross_sections,
                    file_path=csv_roughness_ini_file,
                    fmt="dflow1d",
                    roughness_section=section_file_key_dict[section][1],
                )

        except Exception as e_info:
            self.set_logger_message(
                "An error was produced while exporting files to DIMR format,"
                " not all output files might be exported. "
                f"{e_info!s}",
                level="error",
            )

        # Eport SOBEK 3 format
        try:
            # Cross-sections
            output_exporter.export_geometry(
                cross_sections,
                file_path=csv_geometry_file,
                fmt="sobek3",
            )

            # Roughness
            output_exporter.export_roughness(
                cross_sections,
                file_path=csv_roughness_file,
                fmt="sobek3",
            )
        except Exception as e_info:
            self.set_logger_message(
                "An error was produced while exporting files to SOBEK format,"
                " not all output files might be exported. "
                f"{e_info!s}",
                level="error",
            )

        # Other files:
        try:
            output_exporter.export_geometry(
                cross_sections,
                file_path=csv_geometry_test_file,
                fmt="testformat",
            )

            output_exporter.export_volumes(cross_sections, file_path=csv_volumes_file)
        except Exception as e_info:
            self.set_logger_message(
                "An error was produced while exporting files,"
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
                "Exception thrown while trying to reduce the css points. " + f"{e_message}",
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
                f"Exception thrown while trying to calculate the correction. {e_message}",
                "error",
            )
        return css

    def _print_log_report(self) -> None:
        ll = self.get_logformatter()._loglibrary
        self.set_logger_message(f"Warnings: {ll.get('WARNING')}")
        self.set_logger_message(f"Errors: {ll.get('ERROR')}")

    def _output_exists(self) -> bool:
        """Check whether output exists."""
        for output_file in self._output_files:
            if self.get_inifile().get_output_directory().joinpath(output_file).is_file():
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

    def get_output_directory(self) -> str:
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
