"""
Contains CrossSection class
"""
# Imports from standard library
from datetime import timedelta, datetime
from typing import Mapping, Sequence
from functools import reduce
from time import time
import logging
from logging import Logger
import os

# Imports from dependencies
import numpy as np
import pandas as pd
import scipy.optimize as so
from .lib import polysimplify as PS

# Imports from package
from fm2prof.common import FM2ProfBase, FrictionTable
from fm2prof.MaskOutputFile import MaskOutputFile
from fm2prof import Functions as FE
from fm2prof.IniFile import IniFile
from fm2prof.Import import FmModelData

pd.options.mode.chained_assignment = None  # default='warn'


class CrossSection(FM2ProfBase):
    """
    Use this class to derive cross-sections from fm_data (2D model results).
    See docs how to acquire fm_data and how to prepare a proper 2D model.
    """
    __cs_parameter_css_points = 'MaximumPointsInProfile'
    __cs_parameter_transitionheight_sd = 'SDTransitionHeight'
    __cs_parameter_velocity_threshold = 'AbsoluteVelocityThreshold'
    __cs_parameter_relative_threshold = 'RelativeVelocityThreshold'
    __cs_parameter_min_depth_storage = 'MinimumDepthThreshold'
    __cs_parameter_plassen_timesteps = 'LakeTimesteps'
    __cs_parameter_storagemethod_wli = 'ExtrapolateStorage'
    __cs_parameter_bedlevelcriterium = 'BedlevelCriterium'
    __cs_parameter_SDstorage = 'SDCorrection'
    __cs_parameter_Frictionweighing = 'FrictionweighingMethod'
    __cs_parameter_sdoptimisationmethod = 'sdoptimisationmethod'
    __cs_parameter_skip_maps = 'SkipMaps'
    __cs_parameter_floodplain_base_level = 'SDFloodplainBase'
    __cs_parameter_minwidth = 'MinimumTotalWidth'
    __logger = None

    def __init__(self,
            name: str, length: float, location: tuple,
            branchid="not defined", chainage=0,
            fm_data: FmModelData=None,
            logger: Logger=None, inifile: IniFile=None):
        """
        Arguments:
            InputParam_dict {Dictionary} -- [description]
            name {str} -- [description]
            length {float} -- [description]
            location {tuple} -- [description]

        Keyword Arguments:
            branchid {str} -- [description] (default: {"not defined"})
            chainage {int} -- [description] (default: {0})
        """
        super().__init__(logger=logger, inifile=inifile)

        # Cross-section meta data
        self.name = name                # cross-section id
        self.length = length            # 'vaklengte'
        self.location = location        # (x,y)
        self.branch = branchid          # name of 1D branch for cross-section
        self.chainage = chainage        # offset from beginning of branch
        self.__output_mask_list = []    # initialize output mask list.
        self._fm_data = fm_data         # dictionary with fmdata

        # Cross-section geometry
        self.section_widths = {'main':0, 'floodplain1':0, 'floodplain2':0}
        self.friction_tables = dict()
        self.roughness_sections = np.array([])

        # delta h corrections ("summerdike option")
        self.crest_level = 0
        # in cross-section def. WAQ2PROF did crest - some fixed value.
        #  how to do here?
        self.floodplain_base = 0.0
        # note" 'to avoid numerical oscillation'. might need minimal value.
        # fixed or variable? Test!
        self.transition_height = 0.5
        self.extra_flow_area = 0.0
        self.extra_total_volume = 0.0
        self.extra_area_percentage = list()
        self.extra_total_area = 0
        self.extra_flow_area = 0

        # These attributes are used for non-reduced sets
        self._css_z = 0
        self._css_total_volume = 0
        self._css_total_volume_corrected = None
        self._css_flow_volume = 0
        self._css_flow_volume_corrected = None
        self._css_total_width = 0
        self._css_flow_width = 0
        self._fm_total_volume = 0
        self._fm_flow_volume = 0
        self._fm_wet_area = 0
        self._fm_flow_area = 0
        self._css_volume_legacy = 0

        # flags
        self._css_is_corrected = False
        self._css_is_reduced = False

        # data structures
        self.__output_face_list = []
        self.__output_edge_list = []

        self._section_map = {'0': 'main',
                             '1': 'main',
                             '2': 'floodplain1',
                             '3': 'floodplain2',
                             '-999': 'main',
                             'main': 'main',
                             'floodplain1': 'floodplain1',
                             'floodplain2': 'floodplain2'}

    @property
    def alluvial_width(self):
        for key in [1, '1', 'main', 'Main']:
            try:
                return self.section_widths[key]
            except KeyError:
                pass
        return 0

    @property
    def nonalluvial_width(self):
        for key in [2, '2', 'floodplain', 'FloodPlain1']:
            try:
                return self.section_widths[key]
            except KeyError:
                pass
        return 0

    @property
    def face_points_list(self):
        return self.__output_face_list

    @property
    def edge_points_list(self):
        return self.__output_edge_list

    def get_point_list(self, pointtype):
        if pointtype == 'face':
            return self.face_points_list
        elif pointtype == 'edge':
            return self.edge_points_list
        else:
            raise ValueError('pointtype must be "face" or "edge"')

    # Public functions
    def build_geometry(self):
        """
        Build 1D geometrical cross-section from FM data.

        :param fm_data: dict
        :return:
        """
        fm_data = self._fm_data

        # Unpack FM data
        waterlevel = fm_data['waterlevel'].iloc[:, self.get_parameter(self.__cs_parameter_skip_maps):]
        waterdepth = fm_data['waterdepth'].iloc[:, self.get_parameter(self.__cs_parameter_skip_maps):]
        velocity = fm_data['velocity'].iloc[:, self.get_parameter(self.__cs_parameter_skip_maps):]
        area = fm_data['area']
        bedlevel = fm_data['bedlevel']

        # Convert area to a matrix for matrix operations
        # (much more efficient than for-loops)
        area_matrix = pd.DataFrame(index=area.index)
        for t in waterdepth:
            area_matrix[t] = area

        bedlevel_matrix = pd.DataFrame(index=bedlevel.index)
        for t in waterdepth:
            bedlevel_matrix[t] = bedlevel

        # Retrieve the water-depth
        # & water level nearest to the cross-section location
        self.set_logger_message('Retrieving centre point values')
        (centre_depth, centre_level) = FE.get_centre_values(
            self.location,
            fm_data['x'],
            fm_data['y'],
            waterdepth,
            waterlevel)

        # apply rolling average over the velocities
        # to smooth out extreme values
        velocity = velocity.rolling(
            window=10,
            min_periods=1,
            center=True,
            axis=1).mean()

        # Identify river lakes (plassen)
        self.set_logger_message('Identifying lakes')

        # plassen_mask needed for arrays in output
        (plassen_mask,
            wet_not_plas_mask,
            plassen_depth_correction) = self._identify_lakes(waterdepth)

        # Masks for wet and flow cells (stroomvoeringscriteria)
        self.set_logger_message('Seperating flow from storage')
        flow_mask = self._distinguish_flow_from_storage(waterdepth, velocity)

        # Calculate area and volume as function of waterlevel & waterdepth
        self._fm_wet_area = np.nansum(area_matrix[wet_not_plas_mask], axis=0)
        self._fm_flow_area = np.nansum(area_matrix[flow_mask], axis=0)

        # Correct waterdepth for lakes
        waterdepth = waterdepth + plassen_depth_correction
        waterdepth = waterdepth[waterdepth >= 0]

        # Correct waterdepth for deep pools (volume below deepest point in centre
        # should not be considered to be conveyance)
        pools_id = [i[0] for i in np.argwhere(waterdepth.to_numpy()[:, 0] > centre_depth[0])]
        for pool in pools_id:
            amount_deeper = waterdepth.iloc[pool,0] - centre_depth[0]
            waterdepth.iloc[pool] -= amount_deeper

        #waterdepth[waterdepth > centre_depth[0]] = centre_depth[0]

        # Compute 2D volume as sum of area times depth
        self._fm_total_volume = np.array(
            np.nansum(
                area_matrix[wet_not_plas_mask] * waterdepth[wet_not_plas_mask],
                axis=0))
        self._fm_flow_volume = np.array(
            np.nansum(
                area_matrix[flow_mask] * waterdepth[flow_mask],
                axis=0))

        # For roughness we will need the original z-levels,
        # since geometry z will change below
        self._css_z_roughness = centre_level

        # Check for monotonicity (water levels should rise)
        mono_mask = self._check_monotonicity(centre_level, method=1)
        centre_level = centre_level[mono_mask]
        self._fm_total_volume = self._fm_total_volume[mono_mask]
        self._fm_flow_volume = self._fm_flow_volume[mono_mask]
        self._fm_wet_area = self._fm_wet_area[mono_mask]
        self._fm_flow_area = self._fm_flow_area[mono_mask]

        # Compute geometry above z0 - Water level dependent calculation
        self.set_logger_message('Computing cross-section from water levels')
        self._compute_css_above_z0(centre_level)

        # Compute geometry below z0 - Water level independent calculation
        self.set_logger_message('Computing cross-section from bed levels')
        self._extend_css_below_z0(
            centre_level,
            centre_depth,
            bedlevel_matrix,
            area_matrix,
            plassen_mask)

        # Compute 1D volume as integral of width with respect to z times length
        self._css_total_volume = np.append(
            [0],
            np.cumsum(
                self._css_total_width[1:]*np.diff(self._css_z)*self.length))
        self._css_flow_volume = np.append(
            [0],
            np.cumsum(
                self._css_flow_width[1:]*np.diff(self._css_z)*self.length))

        # If sd correction is run, these attributed will be updated.
        self._css_total_volume_corrected = self._css_total_volume
        self._css_flow_volume_corrected = self._css_flow_volume

        # convert to float64 array for uniformity
        # (apparently entries can be float32)
        self._css_z = np.array(self._css_z, dtype=np.dtype('float64'))

        fm_data['islake'] = plassen_mask

        # generate all mask points for the given cross_section
        #self.set_mask_output_list(fm_data, plassen_mask)

    def check_requirements(self):
        """
        Performs check on cross-section such that it
        hold up to requirements.
        """
        # Remove multiple zeroes in the bottom of the cross-section
        self._check_remove_duplicate_zeroes()
        self._check_remove_zero_widths()

        # Check if cross-sections are in increasing order
        self._css_z = self._check_increasing_order(self._css_z)
        self._css_total_width = self._check_increasing_order(self._css_total_width)
        self._css_flow_width = self._check_increasing_order(self._css_flow_width)

    def calculate_correction(self, transition_height: float):
        """
        Function to determine delta-h correction
        (previously known as 'summerdike option').
        Optimises values for transition height, crest levels and added volume.

        Updates variables:
        self._css_total_volume_corrected

        TODO: to avoid numerical oscillation,
        transition_height need minimal value. Fixed or variable? Test!
        :return:
        """

        # Set initial values for optimisation of parameters
        initial_total_error = self._css_total_volume - self._fm_total_volume
        initial_flow_error = self._css_flow_volume - self._fm_flow_volume

        initial_crest = self._css_z[np.nanargmin(initial_total_error)]
        initial_total_volume = np.abs(initial_total_error[-1])
        initial_flow_volume = np.abs(initial_flow_error[-1])

        self.set_logger_message(
            "Initial crest: {:.4f} m".format(initial_crest), level='debug')
        self.set_logger_message(
            "Initial extra total area: {:.4f} m2".format(initial_total_volume/self.length), level='debug')
        self.set_logger_message(
            "Initial extra flow area: {:.4f} m2".format(initial_flow_volume/self.length), level='debug')

        # Optimise attributes
        opt = self._optimize_sd_storage(initial_crest=initial_crest,
                                        initial_total_volume=initial_total_volume,
                                        initial_flow_volume=initial_flow_volume,
                                        )

        # Unpack optimisation results
        transition_height = self.get_parameter(self.__cs_parameter_transitionheight_sd)
        crest_level = opt.get('crest_level')
        extra_total_volume = opt.get('extra_total_volume')
        extra_flow_volume = opt.get('extra_flow_volume')

        self.set_logger_message(
            "final costs: {:.2f}".format(opt.get('final_cost')), level='debug')
        self.set_logger_message(
            "Optimizer msg: {}".format(opt.get('message')), level='debug')
        self.set_logger_message(
            "Final crest: {:.4f} m".format(crest_level), level='debug')
        self.set_logger_message(
            "Final total area: {:.4f} m2".format(extra_total_volume/self.length), level='debug')
        self.set_logger_message(
            "Final flow area: {:.4f} m2".format(extra_flow_volume/self.length), level='debug')

        extra_area_percentage = FE.get_extra_total_area(
            self._css_z,
            crest_level,
            transition_height)

        # Write to attributes
        self._css_total_volume_corrected = (
            self._css_total_volume + extra_area_percentage * extra_total_volume)
        self._css_flow_volume_corrected = (
            self._css_flow_volume + extra_area_percentage * extra_flow_volume)
        self.crest_level = crest_level
        self.transition_height = transition_height
        self.extra_total_volume = extra_total_volume
        self.extra_flow_volume = extra_flow_volume
        self.extra_total_area = extra_total_volume / self.length
        self.extra_flow_area = extra_flow_volume / self.length
        self.extra_area_percentage = extra_area_percentage
        self._css_is_corrected = True

    def assign_roughness(self) -> None:
        """
        This function builds a table of Chezy values as function of water level
        The roughnes is divides into two sections on the assumption of
        an alluvial (smooth) and nonalluvial (rough) part of the total
        cross-section. This division is made based on the final timestep.
        """

        # Compute roughness tabels
        self.set_logger_message('Building roughness table', 'debug')
        self._build_roughness_tables()
        # Compute roughness widths
        self.set_logger_message('Computing section widths', 'debug')
        self._compute_section_widths()
        # Compute floodplain base level
        self.set_logger_message('Computing base level', 'debug')
        self._compute_floodplain_base()
        # done

    def get_number_of_faces(self) -> int:
        """ use this method to return the number of 2D faces within control volume """
        return len(self._fm_data.get('x'))

    def get_number_of_vertices(self) -> int:
        """ Use this method to return the current number of geometry vertices """
        return len(self._css_total_width)

    def reduce_points(self, count_after: int, method='visvalingam_whyatt', verbose=True):
        """
        Reduces the number of points from _css attributes to a preset maximum.

        Implemented vertex reduction methods:

        'visvalingam_whyatt'
        Based on Visvalingam, M and Whyatt J D (1993),
        "Line Generalisation by Repeated Elimination of Points",
        Cartographic J., 30 (1), 46 - 51

        :param n: int
        :param method: str
        :param verbose: boolean
        :return:
        """

        n_before_reduction = self.get_number_of_vertices()

        points = np.array(
            [
                [self._css_z[i], self._css_total_width[i]]
                for i in range(n_before_reduction)])

        # The number of points is equal to n, it cannot be further reduced
        reduced_index = np.array([True] * n_before_reduction)

        if n_before_reduction != count_after:
            # default is the same value as it came
            if method.lower() == 'visvalingam_whyatt':
                try:
                    simplifier = PS.VWSimplifier(points)
                    reduced_index = simplifier.from_number_index(count_after)
                except Exception as e:
                    self.set_logger_message(
                        'Exception thrown while using polysimplify: ' +
                        '{}'.format(str(e)), 'error')

        # Write to attributes
        self._css_z = self._css_z[reduced_index]
        self._css_total_width = self._css_total_width[reduced_index]
        self._css_flow_width = self._css_flow_width[reduced_index]

        self.set_logger_message(
            'Cross-section reduced ' +
            'from {} '.format(n_before_reduction) +
            'to {} points'.format(len(self.total_width)))

        self._css_is_reduced = True

    def set_face_output_list(self):
        """Generates a list of output mask points based on
        their values in the mask.

        writes to self.__output_mask_list
        Arguments:
            fm_data {dict} -- Dictionary containing x,y values.
            mask_array {NP.array} -- Array of values.
        """
        fm_data = self._fm_data

        # Properties keys
        cross_section_id_key = 'cross_section_id'
        cross_section_region_key = 'region'
        is_lake_key = 'is_lake'
        bedlevel_key = 'bedlevel'
        section_key = 'section'
        region_key = 'region'

        try:
            # Normalize np arrays to list for correct access
            x_coords = fm_data.get('x').tolist()
            y_coords = fm_data.get('y').tolist()
            region_list = fm_data.get('region').tolist()
            section_list = fm_data.get('section').tolist()
            bedlevel_list = fm_data.get('bedlevel').tolist()
            is_lake_mask_list = fm_data.get('islake').tolist()

            # Assume same length for x and y coords.
            for i in range(len(x_coords)):
                mask_properties = {
                    cross_section_id_key: self.name,
                    is_lake_key: is_lake_mask_list[i],
                    cross_section_region_key: region_list[i],
                    bedlevel_key: bedlevel_list[i],
                    region_key: region_list[i],
                    section_key: section_list[i]
                }
                mask_coords = (x_coords[i], y_coords[i])
                # Create the actual geojson element.
                output_mask = MaskOutputFile.create_mask_point(
                    mask_coords,
                    mask_properties)

                if output_mask.is_valid:
                    self.__output_face_list.append(output_mask)
                    #self.set_logger_message(
                    #    'Added output mask at {} '.format(mask_coords) +
                    #    'for Cross Section {}.'.format(self.name),
                    #     level='debug')
                else:
                    self.set_logger_message(
                        'Invalid output mask at {} '.format(mask_coords) +
                        'for Cross Section {}, not added. '.format(self.name) +
                        'Reason {}'.format(output_mask.errors()),
                        level='error')
        except Exception as e_error:
            self.set_logger_message(
                'Error setting output masks ' +
                'for Cross Section {}. '.format(self.name) +
                'Reason: {}'.format(str(e_error)),
                level='error')

    def set_edge_output_list(self):
        """Generates a list of output mask points based on
        their values in the mask.

        writes to self.__output_mask_list
        Arguments:
            fm_data {dict} -- Dictionary containing x,y values.
            mask_array {NP.array} -- Array of values.
        """
        fm_data = self._fm_data

        # Properties keys
        cross_section_id_key = 'cross_section_id'
        cross_section_region_key = 'region'
        roughness_section_key = 'section'

        try:
            # Normalize np arrays to list for correct access
            x_coords = fm_data.get('edge_x').tolist()
            y_coords = fm_data.get('edge_y').tolist()
            section_list = fm_data.get('edge_section').tolist()
            # Assume same length for x and y coords.
            for i in range(len(x_coords)):
                mask_properties = {
                    cross_section_id_key: self.name,
                    roughness_section_key: section_list[i]
                }
                mask_coords = (x_coords[i], y_coords[i])
                # Create the actual geojson element.
                output_mask = MaskOutputFile.create_mask_point(
                    mask_coords,
                    mask_properties)

                if output_mask.is_valid:
                    self.__output_edge_list.append(output_mask)
                else:
                    self.set_logger_message(
                        'Invalid output mask at {} '.format(mask_coords) +
                        'for Cross Section {}, not added. '.format(self.name) +
                        'Reason {}'.format(output_mask.errors()),
                        level='error')
        except Exception as e_error:
            self.set_logger_message(
                'Error setting output masks ' +
                'for Cross Section {}. '.format(self.name) +
                'Reason: {}'.format(str(e_error)),
                level='error')

    def interpolate_roughness_table(self, tablename, z_values):
        """
        Interpolates the roughness table to z values
        """
        table_name_list = ['main', 'floodplain1']

        if tablename not in table_name_list:
            raise KeyError("tablename not in list {}".format(table_name_list))

        pass

    def _check_remove_duplicate_zeroes(self):
        """
        Removes duplicate zeroes in the total width
        """

        # Remove multiple 0s in the total width
        index_of_first_nonzero = max(1, np.argwhere(self._css_total_width!=0)[0][0])

        self._css_z = self._return_first_item_and_after_index(self._css_z, index_of_first_nonzero)
        self._css_flow_width = self._return_first_item_and_after_index(self._css_flow_width, index_of_first_nonzero)
        self._css_total_width = self._return_first_item_and_after_index(self._css_total_width, index_of_first_nonzero)
        self.set_logger_message(f'Removed {index_of_first_nonzero-1} duplicate zero widths', 'debug')
    
    @property
    def z(self):
        return self._css_z

    @property
    def total_width(self):
        return self._css_total_width

    @property
    def flow_width(self):
        return self._css_flow_width

    @staticmethod
    def _return_first_item_and_after_index(listin, after_index):
        return np.append(listin[0], listin[after_index:].tolist())
    
    def _check_remove_zero_widths(self):
        """
        A zero width may lead to numerical instability
        """
        minwidth = self.get_parameter(self.__cs_parameter_minwidth)
        # Replace 0 total width from the first row
        if self._css_total_width[0] == 0:
            if self._css_total_width[1] >= minwidth:
                # minimum of 1.0
                self._css_total_width[0] = minwidth
                self.set_logger_message(f'Set minimum total width to {minwidth}', 'debug')
            else:
                # 0.1m less than the second row (arbitrary)
                self._css_total_width[0] = self._css_total_width[1]-0.1
                self.set_logger_message(f'Set minimum total width to {self._css_total_width[1]-0.1}', 'debug')

    def _combined_optimisation_func(self, opt_in):
        (crest_level, extra_total_volume, extra_flow_volume) = opt_in
        transition_height = self.get_parameter(
            self.__cs_parameter_transitionheight_sd)

        predicted_total_volume = (
            self._css_total_volume +
            FE.get_extra_total_area(
                self._css_z,
                crest_level,
                transition_height) * extra_total_volume)

        predicted_flow_volume = (
            self._css_flow_volume +
            FE.get_extra_total_area(
                self._css_z,
                crest_level,
                transition_height) * extra_flow_volume)

        return FE.return_volume_error(predicted_total_volume + predicted_flow_volume,
                                      self._fm_total_volume + self._fm_flow_volume)

    def _optimisation_func(self, opt_in, *args):
        """
        Objective function used in optimising a delta-h correction
        for parameters:
            crest_level         : level at which the correction begins
            transition_height   : height over which volume is released
            extra_volume        : total extra volume


        :param opt_in: tuple
        :return:
        """
        if args[0][0] == 'both':
            (crest_level, extra_volume) = opt_in

        else:
            (extra_volume) = opt_in
            crest_level = args[0][2]

        volume = args[0][1]
        transition_height = self.get_parameter(
            self.__cs_parameter_transitionheight_sd)

        predicted_volume = (
            volume +
            FE.get_extra_total_area(
                self._css_z,
                crest_level,
                transition_height) * extra_volume)
        return FE.return_volume_error(predicted_volume, self._fm_total_volume)

    def _optimize_sd_storage(self, initial_crest, initial_total_volume, initial_flow_volume):

        # Default option
        sdoptimisationmethod = self.get_parameter(self.__cs_parameter_sdoptimisationmethod)
        if sdoptimisationmethod not in [0, 1, 2]:
            # this should be handled in inifile instead
            self.set_logger_message('sdoptimisationmethod is {} but should be 0, 1, or 2. Defaulting to 0'.format(sdoptimisationmethod),
            level='warning')
            sdoptimisationmethod = 0

        if sdoptimisationmethod == 0:
            self.set_logger_message('Optimising SD on total volume', level='debug')

            # Optimise crest on total volume
            opt = so.minimize(self._optimisation_func,
                            (initial_crest, initial_total_volume),
                            args=['both', self._css_total_volume],
                            method='Nelder-Mead',
                            tol=1e-6)


            crest_level = opt['x'][0]
            extra_total_volume = np.max([opt['x'][1], 0])

            # Optimise flow volume
            opt2 = so.minimize(self._optimisation_func,
                            (initial_flow_volume),
                            args=['notboth', self._css_flow_volume, crest_level],
                            method='Nelder-Mead',
                            tol=1e-6)
            extra_flow_volume = np.min([np.max([opt2['x'][0], 0]), extra_total_volume])

        elif self.get_parameter(self.__cs_parameter_sdoptimisationmethod) == 1:
            self.set_logger_message('Optimising SD on flow volume', level='debug')

            # Optimise crest on flow volume
            opt = so.minimize(self._optimisation_func,
                            (initial_crest, initial_total_volume),
                            args=['both', self._css_flow_volume],
                            method='Nelder-Mead',
                            tol=1e-6)
            crest_level = opt['x'][0]
            extra_flow_volume = np.max([opt['x'][1], 0])

            # Optimise total volume
            opt2= so.minimize(self._optimisation_func,
                            (initial_flow_volume),
                            args=['notboth', self._css_total_volume, crest_level],
                            method='Nelder-Mead',
                            tol=1e-6)
            extra_total_volume = np.max([np.max([opt2['x'][0], 0]), extra_flow_volume])

        elif self.get_parameter(self.__cs_parameter_sdoptimisationmethod) == 2:
            self.set_logger_message('Optimising SD on both flow and total volumes', level='debug')
            opt = so.minimize(self._combined_optimisation_func,
                            (initial_crest, initial_total_volume, initial_flow_volume),
                            method='Nelder-Mead',
                            tol=1e-6)

            crest_level = opt['x'][0]
            extra_total_volume = np.max([opt['x'][1], 0])
            extra_flow_volume = np.min([np.max([opt['x'][2], 0]), extra_total_volume])

        return {'crest_level': crest_level,
                'extra_total_volume': extra_total_volume,
                'extra_flow_volume': extra_flow_volume,
                'final_cost': opt['fun'],
                'message': opt['message']}

    def _check_increasing_order(self, list_points):
        """ there must not be the same values; if so, assign +1mm """
        for indx in range(1,len(list_points)):
            if list_points[indx] == list_points[indx-1]:
                list_points[indx] = list_points[indx] + 0.001
        return list_points

    def _build_roughness_tables(self):

        # Find roughness tables for each section
        chezy_fm = self._fm_data.get('chezy').iloc[:, self.get_parameter(self.__cs_parameter_skip_maps):]

        sections = np.unique(self._fm_data.get('edge_section'))

        for section in sections:
            chezy_section = chezy_fm[self._fm_data['edge_section']==section]
            if self.get_parameter(self.__cs_parameter_Frictionweighing) == 0:
                friction = self._friction_weighing_simple(chezy_section, section)
            elif self.get_parameter(self.__cs_parameter_Frictionweighing) == 1:
                friction = self._friction_weighing_area(chezy_section, section)
            else:
                raise ValueError("unknown option for roughness weighing: {}".format(self.get_parameter(self.__cs_parameter_Frictionweighing)))

            self.friction_tables[self._section_map[str(section)]] = FrictionTable(level=self._css_z_roughness,
                                                              friction=friction)

    def _friction_weighing_simple(self, link_chezy, section):
        """ Simple mean, no weight """
        # Remove chezy where zero
        link_chezy = link_chezy.replace(0, np.NaN)
        output = link_chezy.mean(axis=0).replace(np.NaN, 0)

        return output.values

    def _friction_weighing_area(self, link_chezy, section):
        """
        Compute chezy by weighted average. Weights are determined based on area.

        Friction values are known at flow links, while areas are known at flow faces.

        The area of a flow link is defined as the average of the two faces it connects.

        """
        # Remove chezy where zero
        link_chezy = link_chezy.replace(0, np.NaN)
        efs = self._fm_data['edge_faces'][self._fm_data['edge_section']==section]
        link_area = [self._fm_data.get('area_full')[ef].mean() for ef in efs]
        link_weight = link_area / np.sum(link_area)

        output = np.sum(link_chezy.values.T * link_weight, axis=1)
        output[output==np.NaN] = 0
        return output

    def _compute_section_widths(self):
        for section in np.unique(self._fm_data['edge_section']):
            self.section_widths[self._section_map[str(section)]] = np.sum(self._fm_data['area'][self._fm_data['section']==section])/self.length
            #self.section_widths[section] = self._calc_roughness_width(self._fm_data['edge_section']==section)

    def _compute_floodplain_base(self) -> None:
        """
        Sets the self.floodplain_base attribute. The floodplain
        will be set at least 0.5 meter below the crest of the
        embankment, and otherwise at the average hight of the floodplain
        """

        # Mean bed level in section 2 (floodplain)
        mean_floodplain_elevation = np.mean(self._fm_data['bedlevel'][self._fm_data.get('section') == 2])

        # Tolerance. Base level must at least be some below the crest to prevent
        # numerical issues
        tolerance = self.get_inifile().get_parameter('sdfloodplainbase')
        if (self.crest_level - tolerance) < mean_floodplain_elevation:
            self.floodplain_base = self.crest_level - tolerance
            self.set_logger_message(f'Mean floodpl. elev. ({mean_floodplain_elevation:.2f} m)'+
                                    f'higher than crest level ({self.crest_level:.2f}) + '+
                                    f'tolerance ({tolerance} m)', 'warning')
        else:
            self.floodplain_base = mean_floodplain_elevation
            self.set_logger_message(f'Floodplain base level set to {mean_floodplain_elevation:.2f} m', 'debug')

    def _calc_roughness_width(self, link_indices):
        # get the 2 nodes for every alluvial edge
        fm_data = self._fm_data

        section_nodes = fm_data['edge_nodes'][link_indices]
        section_nodes = np.unique(section_nodes.flatten())

        # get the faces for every node (start at index 1)
        faces = np.array(
            [index
                for index, nodes in enumerate(fm_data['face_nodes_full'])
                for node in nodes
                if node in section_nodes])
        faces += 1

        (unique_faces, faces_count) = np.unique(faces, return_counts=True)

        # only faces that occur at least 3 times (2 edges, 3 nodes)
        # belong to the section
        # this approach works for both square as well as triangular meshes
        section_faces = unique_faces[faces_count >= 3]

        mask = np.array(
            [index in section_faces
                for index in range(1, fm_data['area_full'].size + 1)])
        section_width = np.sum(fm_data['area_full'][mask])/self.length

        return section_width

    def _calc_chezy(self, depth, manning):
        return depth**(1/float(6)) / manning

    def _identify_lakes(self, waterdepth):
        """
        Find cells that do not have rising water depths
        for a period of time (__cs_parameter_plassen_timesteps)

        :param waterdepth: waterdepths from fm_data

        :return plassen_mask: mask of all cells that contain a lake
        :return wet_not_plas_mask: mask of all cells that are wet,
            but not a lake
        :return plassen_depth_correction: the depth of lake
            at the start of a computation
        """
        # check for non-rising waterlevels
        waterdepth_diff = np.diff(waterdepth, n=1, axis=-1)

        # this creates a series
        # for waal, 2 steps = 32 hours
        plassen_mask = (
            waterdepth.T.iloc[
                self.get_parameter(
                    self.__cs_parameter_plassen_timesteps)] > 0) & \
            (np.abs(
                waterdepth.T.iloc[
                    self.get_parameter(
                        self.__cs_parameter_plassen_timesteps
            )] - waterdepth.T.iloc[0]) <= 0.01)

        self.plassen_mask = plassen_mask

        plassen_mask_time = np.zeros(
            (len(waterdepth.T), len(plassen_mask)),
            dtype=bool)
        plassen_mask_time[0, :] = plassen_mask

        plassen_depth_correction = np.zeros(waterdepth.shape, dtype=float)

        for i, depths in enumerate(waterdepth):
            plassen_depth_correction[plassen_mask, i] = \
                -waterdepth.T.iloc[0][plassen_mask]

        # walk through dataframe in time, for each timestep check
        # when to unmask a plassen cell
        for i, diff in enumerate(waterdepth_diff.T):
            final_mask = reduce(
                np.logical_and,
                [
                    (diff <= 0.001),
                    (plassen_mask_time[i] == True)])
            plassen_mask_time[i + 1, :] = final_mask

        plassen_mask_time = pd.DataFrame(plassen_mask_time).T

        # find all wet cells
        wet_mask = waterdepth > 0

        # correct wet cells for plassen
        wet_not_plas_mask = reduce(
            np.logical_and,
            [
                (wet_mask == True),
                (plassen_mask_time == False)])
        # print (wet_not_plas_mask)
        # pmtmask = plassen_mask_time == False
        # print (wet_mask & pmtmask)
        return plassen_mask, wet_not_plas_mask, plassen_depth_correction

    def _compute_css_above_z0(self, centre_level):
        """
        'Area method': compute width from area

        """

        self._css_z = centre_level

        # Remove nan's from z
        self._css_z[np.isnan(self._css_z)] = np.nanmin(self._css_z)

        # Compute widths
        self._css_total_width = np.array(self._fm_wet_area)/self.length
        self._css_flow_width = np.array(self._fm_flow_area)/self.length

        # Flow width must increase at each z
        for i in range(2, len(self._css_flow_width)+1):
            self._css_flow_width[-i] = np.min(
                [self._css_flow_width[-i], self._css_flow_width[-i+1]])

        # fix error when flow_width > total_width (due to floating points)
        self._css_flow_width[
            self._css_flow_width >
            self._css_total_width] = self._css_total_width[
                self._css_flow_width > self._css_total_width]

    def _distinguish_flow_from_storage(self, waterdepth, velocity):
        """
        Defines mask which is False when flow is below a certain threshold.
        (Stroomvoeringscriteriumfunctie)

        Arguments:
            waterdepth: pandas Dataframe with cell id
                for index and time in columns
            velocity: pandas Dataframe with cell id
                for index and time in columns

        Returns:
            flow_mask: pandas Dataframe with cell id
                for index and time in columns.
                True for flow, False for storage
        """
        flow_mask = (waterdepth > 0) & \
                    (velocity > self.get_parameter(
                        self.__cs_parameter_velocity_threshold)) & \
                    (velocity > self.get_parameter(
                        self.__cs_parameter_relative_threshold
                     ) * np.mean(velocity))

        # shallow flows should not be used for identifying storage
        # (cells with small velocities are uncorrectly seen as storage)
        waterdepth_correction = (waterdepth > 0) & \
            (waterdepth < self.get_parameter(
                self.__cs_parameter_min_depth_storage))

        # combine flow and depth mask to avoid assigning shallow flows
        # to storage
        # flow_mask = reduce(
        # np.logical_or,
        # [(flow_mask == True), (waterdepth_correction == True)])
        flow_mask = flow_mask | waterdepth_correction

        return flow_mask

    def _extend_css_below_z0(
            self,
            centre_level, centre_depth,
            bedlevel_matrix,
            area_matrix,
            plassen_mask):
        """
        Extends the cross-sectional information below the water level
        at the first timestep,
        under assumption that the polygon formed by the water level
        and the bed level is convex.
        It works by walking down to the bed level at the center point
        in 'virtual water level steps'.
        At each step, we sum the area of cells width bedlevels which
        would be submerged at that virtual water level.
        """
        filter_by_depth_percentage = self.get_parameter(self.__cs_parameter_bedlevelcriterium) * 100
        filter_value = FE.empirical_ppf([filter_by_depth_percentage], bedlevel_matrix.values.T[0])[0]
        self.set_logger_message("Lowest {}% of bed levels are filtered (z<{:.4f}m)".format(filter_by_depth_percentage,
                                                                                         filter_value),
                                 level='debug')
        level_z0 = centre_level[0]
        bdata = bedlevel_matrix[~plassen_mask]
        bmask = (bdata < level_z0) & (bdata >= filter_value)

        self.set_logger_message("Number of points below z0 after applying filter: {}".format(np.sum(bmask.values.T[0])),
                                  level='debug')

        # Compute values at z0
        bedlevels_below_z0 = bdata[bmask]
        lowest_level_below_z0 = centre_level[0] - centre_depth[0]
        flow_area_at_z0 = self._fm_flow_area[0]
        total_area_at_z0 = self._fm_wet_area[0]

        for unique_level_below_z0 in reversed(
                np.linspace(lowest_level_below_z0, level_z0, 10)):

            # count area
            areas = area_matrix[bedlevels_below_z0 <= unique_level_below_z0]

            # Set area, such that the width computed from this area is equal or lower
            # than the minimum width from the flow-dependent level
            area_at_unique_level = np.min([np.nansum(areas.values.T[-1]),
                                           total_area_at_z0])

            # Extension of flow/storage below z0
            if self.get_parameter(self.__cs_parameter_storagemethod_wli) == 0:
                flow_area_at_unique_level = area_at_unique_level
            elif self.get_parameter(self.__cs_parameter_storagemethod_wli) == 1:
                flow_area_at_unique_level = np.min([area_at_unique_level, flow_area_at_z0])

            # Insert values in existing arrays
            self._css_z = np.insert(self._css_z, 0, unique_level_below_z0)
            self._css_flow_width = np.insert(
                self._css_flow_width,
                0,
                flow_area_at_unique_level/self.length)
            self._css_total_width = np.insert(
                self._css_total_width,
                0,
                area_at_unique_level/self.length)
            self._fm_wet_area = np.insert(
                self._fm_wet_area,
                0,
                area_at_unique_level)
            self._fm_flow_area = np.insert(
                self._fm_flow_area,
                0,
                flow_area_at_unique_level)
            self._fm_flow_volume = np.insert(self._fm_flow_volume, 0, np.nan)
            self._fm_total_volume = np.insert(self._fm_total_volume, 0, np.nan)

    @staticmethod
    def _check_monotonicity(arr, method=2):
        """
        for given input array, create mask such that when applied to the array,
        all values are monotonically rising

        method 1: remove values were z is falling from array
        method 2: sort array such that z is always rising (default)

        Arguments:
            arr: 1d numpy array

        return:
            mask such that arr[mask] is monotonically rising
        """
        if method == 1:
            mask = np.array([True])
            for i in range(1, len(arr)):
                # Last index that had rising value
                j = np.argwhere(mask)[-1][0]
                if arr[i] > arr[j]:
                    mask = np.append(mask, True)
                else:
                    mask = np.append(mask, False)

            return mask
        elif method == 2:
            return np.argsort(arr)

    def get_parameter(self, key: str):
        return self.get_inifile().get_parameter(key)
