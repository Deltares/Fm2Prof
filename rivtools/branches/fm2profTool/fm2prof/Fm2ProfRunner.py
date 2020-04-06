from fm2prof.common import FM2ProfBase, FmModelData
from fm2prof.CrossSection import CrossSection
from fm2prof import Functions as FE

from fm2prof import sobek_export
from fm2prof import IniFile
from fm2prof.MaskOutputFile import MaskOutputFile
from fm2prof.RegionPolygonFile import RegionPolygonFile, SectionPolygonFile

from typing import Mapping, Sequence
import datetime
import itertools
import os
import shutil
import logging
import time
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import ConvexHull
from geojson import Polygon, Feature, FeatureCollection
import geojson


class Fm2ProfRunner(FM2ProfBase):
    __map_file_key = 'fm_netcdfile'
    __css_file_key = 'crosssectionlocationfile'
    __region_file_key = 'regionpolygonfile'
    __section_file_key = 'sectionpolygonfile'
    __export_mapfiles_key = "exportmapfiles"
    __css_selection_key = "cssselection"
    __classificationmethod_key = "classificationmethod"

    def __init__(self, iniFilePath: str, version: float = None):
        """
        Initializes the private variables for the Fm2ProfRunner

        Arguments:
            iniFilePath {str}
                -- File path where the IniFile is located.
            version {float}
                -- Current version of the software, needs to be rethought.
        """
        self.__iniFile = IniFile.IniFile(iniFilePath)
        self.__version = version

        self._create_logger()

    def run(self):
        """
        Runs the Fm2Prof functionality.
        """
        if self.__iniFile is None:
            self.set_logger_message(
                'No ini file was specified and the run cannot go further.')
            return
        self.run_inifile(self.__iniFile)

    def run_inifile(self, iniFile: IniFile):
        """Runs the desired emulation from 2d to 1d given the mapfile
            and the cross section file.

        Arguments:
            iniFile {IniFile}
                -- Object containing all the information
                    needed to execute the program
        """
        if not self.__is_output_directory_set(iniFile):
            return

        # shorter local variables
        map_file = iniFile._input_file_paths.get(self.__map_file_key)
        css_file = iniFile._input_file_paths.get(self.__css_file_key)
        region_file = iniFile._input_file_paths.get(self.__region_file_key)
        section_file = iniFile._input_file_paths.get(self.__section_file_key)
        output_dir = iniFile._output_dir
        input_param_dict = iniFile._input_parameters

        # Add a log file
        self.set_logfile(
            output_dir=output_dir,
            filename='fm2prof.log')
        self.set_logger_message(
            'FM2PROF version {}'.format(self.__version__))
        self.set_logger_message('reading FM and cross-sectional data data')

        # Read region polygon
        regions = None
        sections = None
        input_param_dict['sectionsmethod'] = 0

        if region_file:
            regions = RegionPolygonFile(region_file, logger=self.get_logger())

        if section_file:
            sections = SectionPolygonFile(section_file, logger=self.get_logger())
            input_param_dict['sectionsmethod'] = 1

        # Read FM model data
        fm2prof_fm_model_data = \
            self._set_fm_model_data(map_file, css_file, regions, sections)
        fm_model_data = FmModelData(fm2prof_fm_model_data)

        ntsteps = fm_model_data.time_dependent_data.get('waterlevel').shape[1]
        nfaces = fm_model_data.time_dependent_data.get('waterlevel').shape[0]
        nedges = fm_model_data.edge_data.get('x').shape[0]
        self.set_logger_message(
            'finished reading FM and cross-sectional data data')
        self.set_logger_message(
          'Number of: timesteps ({}), '.format(ntsteps) +\
          'faces ({}), '.format(nfaces)+\
          'edges ({})'.format(nedges), 
          level='debug')
        # check if edge/face data is available
        if 'edge_faces' not in fm_model_data.edge_data:
            if input_param_dict.get('frictionweighing') == 1:
                self.set_logger_message(
                    'Friction weighing set to 1 (area-weighted average' +
                    'but FM map file does contain the *edge_faces* keyword.' +
                    'Area weighting is not possible. Defaulting to simple unweighted' +
                    'averaging',
                    level='warning')


        # generate all cross-sections
        cross_sections = self._generate_cross_section_list(
            input_param_dict, fm_model_data)

        # The roughness tables in 1D model require
        # the same discharges on the rows.
        # This function interpolates to get the roughnesses
        # at the correct discharges.
        self.start_new_log_task('Finalizing')
        self.set_logger_message(
            'Interpolating roughness')
        FE.interpolate_roughness(cross_sections)

        # Export cross sections
        self.set_logger_message(
            'Export model input files to {}'.format(output_dir))
        self._export_cross_sections(cross_sections, output_dir)

        # Generate output geojson
        try:
            export_mapfiles = input_param_dict[self.__export_mapfiles_key]
        except KeyError:
            # If key is missing, do not export files by default. 
            # We need a better solution for this (inifile.getparam?.. handle defaults there?)
            export_mapfiles = False
        if export_mapfiles:
            self.set_logger_message(
                'Export geojson output to {}'.format(output_dir))
            self._generate_geojson_output(output_dir, cross_sections)
        
        # Export bounding boxes of cross-section control volumes
        try:
            self._export_envelope(output_dir, cross_sections)
        except Exception as e_error:
            e_message = str(e_error)
            self.set_logger_message('Error while exporting bounding boxes', 'error')
            self.set_logger_message(e_message, "error")

    def _export_envelope(self, output_dir, cross_sections):
        """
        # Export envelopes around cross-sections
        """
        output = {"type": "FeatureCollection"}
        css_hulls = list()
        for css in cross_sections:
            pointlist = np.array([point['geometry']['coordinates'] for point in css.get_point_list('face')])
            # construct envelope
            try:
                hull = ConvexHull(pointlist)
                css_hulls.append(Feature(
                    properties= {'name': css.name},
                    geometry=Polygon([list(map(tuple, pointlist[hull.vertices]))])))
            except IndexError:
                self.set_logger_message(f'No Hull Exported For {css.name}')

        with open(os.path.join(output_dir, 'cross_section_volumes.geojson'), 'w') as f: geojson.dump(FeatureCollection(css_hulls), f, indent=2)

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

        # Read FM map file
        self.set_logger_message('Opening FM Map file')
        (time_independent_data, edge_data, node_coordinates, time_dependent_data) = FE._read_fm_model(res_file)
        self.set_logger_message('Closed FM Map file')
        
        # Load locations and names of cross-sections
        self.set_logger_message('Opening css file')
        cssdata = FE._read_css_xyz(css_file)
        self.set_logger_message('Closed css file')

        # Classify regions & set cross-sections
        if (self.__iniFile._input_parameters.get(self.__classificationmethod_key) == 0 ) or (regions is None):
            self.set_logger_message('All 2D points assigned to the same region and classifying points to cross-sections')
            time_independent_data, edge_data = FE.classify_without_regions(cssdata, time_independent_data, edge_data)
        elif self.__iniFile._input_parameters.get(self.__classificationmethod_key) == 1:
            self.set_logger_message('Assigning 2D points to regions using DeltaShell and classifying points to cross-sections')
            time_independent_data, edge_data = self._classify_with_deltashell(time_independent_data, edge_data, cssdata, regions, polytype='region')
        else:
            self.set_logger_message('Assigning 2D points to regions using Built-In method and classifying points to cross-sections')
            time_independent_data, edge_data = self._classify_with_builtin_methods(time_independent_data, edge_data, cssdata, regions)

        # Classify sections for roughness tables
        if (self.__iniFile._input_parameters.get(self.__classificationmethod_key) == 0 ) or (sections is None):
            self.set_logger_message('Assigning point to sections without polygons')
            edge_data = FE.classify_roughness_sections_by_variance(edge_data, time_dependent_data['chezy_edge'])
            time_independent_data = FE.classify_roughness_sections_by_variance(time_independent_data, time_dependent_data['chezy_mean'])
        elif self.__iniFile._input_parameters.get(self.__classificationmethod_key) == 1:
            self.set_logger_message('Assigning 2D points to sections using DeltaShell')
            time_independent_data, edge_data = self._classify_section_with_deltashell(time_independent_data, edge_data)
        else:
            self.set_logger_message('Assigning 2D points to sections using Built-In method')
            edge_data = FE.classify_roughness_sections_by_polygon(sections, edge_data, self.get_logger())
            time_independent_data = FE.classify_roughness_sections_by_polygon(sections, time_independent_data, self.get_logger())

        return time_dependent_data, time_independent_data, edge_data, node_coordinates, cssdata

    def _classify_with_builtin_methods(self, time_independent_data, edge_data, cssdata, polygons):
        # Determine in which region each cross-section lies
            css_regions = polygons.classify_points(cssdata['xy'])

            # Determine in which region each 2d point lies
            xy_tuples_2d = [(time_independent_data.get('x').values[i], 
                     time_independent_data.get('y').values[i]) for i in range(len(time_independent_data.get('x')))]
    
            time_independent_data['region'] = regions.classify_points(xy_tuples_2d)

            xy_tuples_2d = [(edge_data.get('x')[i], 
                            edge_data.get('y')[i]) for i in range(len(edge_data.get('x')))]
            
            edge_data['region'] = polygons.classify_points(xy_tuples_2d)

            # Do Nearest neighbour cross-section for each region
            time_independent_data, edge_data = FE.classify_with_regions(regions, cssdata, time_independent_data, edge_data, css_regions)

            return time_independent_data, edge_data
        
    def _classify_section_with_deltashell(self, time_independent_data, edge_data):

        # Determine in which section each 2D point lies
        self.set_logger_message('Assigning faces...')
        time_independent_data = self._assign_polygon_using_deltashell(time_independent_data, dtype='face', polytype='section')
        self.set_logger_message('Assigning edges...')
        edge_data = self._assign_polygon_using_deltashell(edge_data, dtype='edge', polytype='section')

        
        return time_independent_data, edge_data

    def _classify_with_deltashell(self, time_independent_data, edge_data, cssdata, polygons, polytype='region'):
        
        # Determine in which region each 2D point lies
        self.set_logger_message('Assigning faces...')
        time_independent_data = self._assign_polygon_using_deltashell(time_independent_data, dtype='face', polytype=polytype)
        self.set_logger_message('Assigning edges...')
        edge_data = self._assign_polygon_using_deltashell(edge_data, dtype='edge', polytype=polytype)

        # Determine in which region each cross-section lies
        css_regions = polygons.classify_points(cssdata['xy'])

        # Do Nearest neighbour cross-section for each region
        time_independent_data, edge_data = FE.classify_with_regions(polygons, cssdata, time_independent_data, edge_data, css_regions)

        return time_independent_data, edge_data

    def _get_region_map_file(self, polytype):
        """ Returns the path to a NC file with region ifnormation in the bathymetry data"""
        map_file_path = self.__iniFile._input_file_paths.get(self.__map_file_key)
        filepath, ext = os.path.splitext(map_file_path)
        modified_file_path = f"{filepath}_{polytype.upper()}BATHY{ext}"
        return modified_file_path

    def _assign_polygon_using_deltashell(self, data, dtype: str='face', polytype: str='region'):
        """ Assign all 2D points using DeltaShell method """

        # NOTE
        self.set_logger_message(f'Looking for _{polytype.upper()}BATHY.nc', 'debug')
        
        path_to_modified_nc = self._get_region_map_file(polytype)
        
        # Load Modified NetCDF
        with Dataset(path_to_modified_nc) as nf:
            # Data stored in node z, while fm2prof uses data at faces or edges.
            region_at_node = nf.variables.get('mesh2d_node_z')[:].data.astype(int)

            if dtype == 'face':
                node_to_face = nf.variables.get('mesh2d_face_nodes')[:].data
                region_at_face = region_at_node[node_to_face.T[0]-1]
                data[polytype] = region_at_face
            elif dtype == 'edge':
                node_to_edge = data['edge_nodes']
                region_at_edge = region_at_node[node_to_edge.T[0]-1]
                data[polytype] = region_at_edge
            
        return data

    def _generate_geojson_output(
            self,
            output_dir: str,
            cross_sections: list):
        """Generates geojson file based on cross sections.

        Arguments:
            output_dir {str} -- Output directory path.
            cross_sections {list} -- List of Cross Sections.
        """
        for pointtype in ['face', 'edge']:
            output_file_path = os.path.join(output_dir, '{}_output.geojson'.format(pointtype))
            try:
                node_points = [
                    node_point
                    for cs in cross_sections
                    for node_point in cs.get_point_list(pointtype)]
                self.set_logger_message("Collected points, dumping to file", level='debug')
                MaskOutputFile.write_mask_output_file(
                    output_file_path,
                    node_points)
                self.set_logger_message("Done", level='debug')
            except Exception as e_info:
                self.set_logger_message(
                    'Error while generation .geojson file,' +
                    'at {}'.format(output_file_path) +
                    'Reason: {}'.format(str(e_info)),
                    level='error'
                )

    def _generate_cross_section_list(
            self,
            input_param_dict: Mapping[str, list],
            fm_model_data: FmModelData):
        """ Generates cross sections based on the given fm_model_data

        Arguments:
            input_param_dict {Mapping[str, list]}
                -- Dictionary of parameters read from IniFile
            fm_model_data {FmModelData}
                -- Class with all necessary data for generating Cross Sections

        Returns:
            {list} -- List of generated cross sections
        """
        cross_sections = list()
        if not fm_model_data or not input_param_dict:
            return cross_sections
        
        
        
        # Preprocess css from fm_model_data so it's easier to handle it.
        css_data_list = fm_model_data.css_data_list
        css_selection = self._get_css_range(number_of_css=len(css_data_list))
        self.get_logformatter().set_number_of_iterations(len(css_selection))
        
        for css_data in np.array(css_data_list)[css_selection]:
            generated_cross_section = self._generate_cross_section(
                css_data, input_param_dict, fm_model_data)
            if generated_cross_section is not None:
                cross_sections.append(generated_cross_section)

        return cross_sections

    def _get_css_range(self, number_of_css: int):
        """ parses the CssSelection keyword from the inifile """
        cssSelection = self.__iniFile._input_parameters.get(self.__css_selection_key)
        if cssSelection is None:
            cssSelection = np.arange(0, number_of_css)
        else:
            cssSelection = np.array(cssSelection)
        return cssSelection

    def _generate_cross_section(
            self,
            css_data: dict,
            input_param_dict: Mapping[str, list],
            fm_model_data: FmModelData) -> CrossSection:
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
            raise Exception('No data was given to create a Cross Section')

        css_name = css_data.get('id')
        if not css_name:
            css_name = 'new_cross_section'

        if input_param_dict is None:
            raise Exception(
                'No input parameters (from ini file)' +
                ' given for new cross section {}'.format(css_name))

        if fm_model_data is None:
            raise Exception(
                'No FM data given for new cross section {}'.format(css_name))

        self.start_new_log_task(f"{css_name}")
        # Create cross section
        created_css = self._get_new_cross_section(
            css_data=css_data,
            input_param_dict=input_param_dict)

        if created_css is None:
            raise Exception('No Cross-section could be generated')

        created_css.set_logger(self.get_logger())
        self.set_logger_message(
            'Initiated new cross-section')

        self._set_fm_data_to_cross_section(
            cross_section=created_css,
            input_param_dict=input_param_dict,
            fm_model_data=fm_model_data)

        self.set_logger_message(
            'done')
        return created_css

    def _set_fm_data_to_cross_section(
            self,
            cross_section: CrossSection,
            input_param_dict: Mapping[str, list],
            fm_model_data: FmModelData,
            start_time=None):
        """Sets extra FM data to the given Cross Section

        Arguments:
            cross_section {CrossSection}
                -- Given Cross Section.
            input_param_dict {Mapping[str, list]}
                -- Dictionary with input parameters.
            fm_model_data {FmModelData}
                -- Data to assign to the new cross section.

        """

        if cross_section is None or fm_model_data is None:
            return

        # define fm_model_data variables
        time_independent_data = fm_model_data.time_independent_data
        edge_data = fm_model_data.edge_data
        time_dependent_data = fm_model_data.time_dependent_data
        css_name = cross_section.name
        try:
            # Retrieve FM data for cross-section
            fm_data = FE.get_fm2d_data_for_css(
                cross_section.name,
                time_independent_data,
                edge_data,
                time_dependent_data)

            self.set_logger_message(
                'Retrieved data for cross-section')

            # Build cross-section
            self.set_logger_message('Start building geometry')
            cross_section.build_geometry(fm_data=fm_data)

            self.set_logger_message(
                'Cross-section derived, starting correction.....')

            # Delta-h correction
            self._calculate_css_correction(
                input_param_dict, cross_section)

            # Reduce number of points in cross-section
            self._reduce_css_points(
                input_param_dict, cross_section)

            # assign roughness
            self.set_logger_message('Starting computing roughness tables')
            cross_section.assign_roughness(fm_data)

            self.set_logger_message(
                'computed roughness')

            cross_section.set_face_output_list()
            cross_section.set_edge_output_list()

        except Exception as e_info:
            self.set_logger_message(
                'Exception thrown while setting cross-section' +
                ' {} details, {}'.format(css_name, str(e_info)),
                level='error')

    def _get_new_cross_section(
            self,
            css_data: Mapping[str, str],
            input_param_dict: Mapping[str, str]):
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

        css_data_id = css_data.get('id')
        if not css_data_id:
            return None

        # Get remainig data
        css_data_length = css_data.get('length')
        css_data_location = css_data.get('xy')
        css_data_branch_id = css_data.get('branchid')
        css_data_chainage = css_data.get('chainage')

        if (css_data_length is None or
                css_data_location is None or
                css_data_branch_id is None or
                css_data_chainage is None):
            return None

        try:
            css = CrossSection(
                input_param_dict,
                name=css_data_id,
                length=css_data_length,
                location=css_data_location,
                branchid=css_data_branch_id,
                chainage=css_data_chainage)
        except Exception as e_info:
            self.set_logger_message(
                'Exception thrown while creating cross-section ' +
                '{}, message: {}'.format(css_data_id, str(e_info)))
            return None

        return css

    def _export_cross_sections(self, cross_sections: list, output_dir: str):
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

        # File paths
        css_location_ini_file = os.path.join(
            output_dir, 'CrossSectionLocations.ini')
        css_definitions_ini_file = os.path.join(
            output_dir, 'CrossSectionDefinitions.ini')


        csv_geometry_file = output_dir + '\\geometry.csv'
        csv_roughness_file = output_dir + '\\roughness.csv'

        csv_geometry_test_file = output_dir + '\\geometry_test.csv'
        csv_roughness_test_file = output_dir + '\\roughness_test.csv'

        csv_volumes_file = output_dir + '\\volumes.csv'

        # export fm1D format
        try:
            # Export locations
            sobek_export.export_crossSectionLocations(
                cross_sections, file_path=css_location_ini_file)
            
            # Export definitions
            sobek_export.export_geometry(
                cross_sections,
                file_path=css_definitions_ini_file,
                fmt='dflow1d')

            # Export roughness
            sections = np.unique([s for css in cross_sections for s in css.friction_tables.keys()])
            sectionFileKeyDict = {"main": ['\\roughness-Main.ini', "Main"],
                                  "floodplain1": ['\\roughness-FloodPlain1.ini', "FloodPlain1"],
                                  "floodplain2": ['\\roughness-FloodPlain2.ini', "FloodPlain2"]}
            for section in sections:
                csv_roughness_ini_file = output_dir + sectionFileKeyDict[section][0]
                sobek_export.export_roughness(
                    cross_sections,
                    file_path=csv_roughness_ini_file,
                    fmt='dflow1d',
                    roughness_section=sectionFileKeyDict[section][1])
        except Exception as e_info:
            self.set_logger_message(
                'An error was produced while exporting files to DIMR format,' +
                ' not all output files might be exported. ' +
                '{}'.format(str(e_info)),
                level='error')

        # Eport SOBEK 3 format
        try:
            # Cross-sections
            sobek_export.export_geometry(
                cross_sections, file_path=csv_geometry_file, fmt='sobek3')
            
            # Roughness
            sobek_export.export_roughness(
                cross_sections,
                file_path=csv_roughness_file,
                fmt='sobek3')
        except Exception as e_info:
            self.set_logger_message(
                'An error was produced while exporting files to SOBEK format,' +
                ' not all output files might be exported. ' +
                '{}'.format(str(e_info)),
                level='error')
        
        # Other files:
        try:
            sobek_export.export_geometry(
                cross_sections,
                file_path=csv_geometry_test_file,
                fmt='testformat')

            sobek_export.export_volumes(
                cross_sections,
                file_path=csv_volumes_file)
        except Exception as e_info:
            self.set_logger_message(
                'An error was produced while exporting files,' +
                ' not all output files might be exported. ' +
                '{}'.format(str(e_info)),
                level='error')

        self.set_logger_message('Exported output files, FM2PROF finished')

    def _calculate_css_correction(
            self,
            input_param_dict: Mapping[str, float],
            css: CrossSection):
        """Calculates the Cross Section correction if needed.

        Arguments:
            input_param_dict {Mapping[str,float]} -- [description]
            css {CrossSection} -- [description]

        """
        if not css:
            self.set_logger_message('No Cross Section was provided.')
            return

        # Get value, it should already come as an integer.
        sd_storage_key = 'sdstorage'
        sd_storage_value = input_param_dict.get(sd_storage_key)

        # Get value, it should already come as a float.
        sd_transition_key = 'transitionheight_sd'
        sd_transition_value = input_param_dict.get(sd_transition_key)

        # Check if it should be corrected.
        if sd_storage_value == 1:
            try:
                css.calculate_correction(sd_transition_value)
                self.set_logger_message(
                    'correction finished')
            except Exception as e_error:
                e_message = str(e_error)
                self.set_logger_message(
                    'Exception thrown ' +
                    'while trying to calculate the correction. ' +
                    '{}'.format(e_message))

    def _reduce_css_points(
            self,
            input_param_dict: Mapping[str, str],
            css: CrossSection):
        """Returns a valid value for the number of css points read from ini file.

        Arguments:
            input_param_dict {Mapping[str, str]}
                -- Dictionary of elements read in the Ini File

        Returns:
            {Integer} -- Valid number of css points (default: 20)
        """
        num_css_pt_key = 'number_of_css_points'
        # Will return None if not found / defined.
        num_css_pt_value = input_param_dict.get(num_css_pt_key)
        # 20 is our default value for css_pt_value
        css_pt_value = 20
        try:
            # try to reduce points.
            css_pt_value = int(num_css_pt_value)
        except:
            self.set_logger_message(
                'number_of_css_points given is not an int;' +
                ' Will use default (20) instead')

        try:
            css.reduce_points(n=css_pt_value, verbose=False)

        except Exception as e_error:
            e_message = str(e_error)
            self.set_logger_message(
                'Exception thrown while trying to reduce the css points. ' +
                '{}'.format(e_message))

    

    def __get_time_stamp_seconds(self, start_time: datetime):
        """Returns a time stamp with the time difference

        Arguments:
            start_time {datetime} -- Initial date time

        Returns:
            {float} -- difference of time between start and now in seconds
        """
        time_now = datetime.datetime.now()
        time_difference = time_now - start_time
        return time_difference.total_seconds()

    def __is_output_directory_set(self, iniFile: IniFile):
        """
        Verifies if the output directory has been set and exists or not.
        Arguments:
            iniFile {IniFile} -- [description]
        Returns:
            True - the output_dir is set and exists.
            False - the output_dir is not set or does not exist.
        """
        if iniFile is None or iniFile._output_dir is None:
            print("The output directory must be set before running.")
            return False

        if not os.path.exists(iniFile._output_dir):
            try:
                os.makedirs(iniFile._output_dir)
            except:

                print(
                    'The output directory {}, '.format(iniFile._output_dir) +
                    'could not be found neither created.')
                return False

        return True

    def __generate_output(
            self, output_directory: str, fig, figType: str, name: str):
        if not self.__saveFigures:
            return

        plotLocation = output_directory + '\\{0}_{1}.png'.format(name, figType)
        fig.savefig(plotLocation)
        self.set_logger_message(
            'Saved {} for {} plot in {}.'.format(name, figType, plotLocation))

        return
