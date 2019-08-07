#! /usr/bin/env python
"""
This module contains classes used for the emulation/reduction of 2D models to 1D models for Delft3D FM (D-Hydro).

CrossSection - for 'FM2PROF'

Dependencies
------------------
Packages, between parenthesis known working version.

numpy (1.10.4)
pandas (0.17.1)
matplotlib (1.5.1)

Contact: K.D. Berends (koen.berends@deltares.nl, k.d.berends@utwente.nl)
"""

# region // imports
import numpy as np
import pandas as pd
import logging
from datetime import timedelta, datetime
from functools import reduce
import scipy.optimize as so
from time import time
import seaborn as sns

from fm2prof import common
from fm2prof import Functions as FE
from typing import Mapping, Sequence
from .lib import polysimplify as PS

import os

pd.options.mode.chained_assignment = None  # default='warn'
# endregion

# Module information
__author__ = "Koen Berends"
__copyright__ = "Copyright 2016, University of Twente & Deltares"
__credits__ = ["Koen Berends"]
__license__ = "no license (restricted)"
__version__ = "$Revision$"
__maintainer__ = "Koen Berends"
__email__ = "koen.berends@deltares.nl"
__status__ = "Prototype"


class FmModelData:
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
    def get_ordered_css_list(css_data_dict: Mapping[str,str]):
        """Returns an ordered list where every element represents a Cross Section structure

        Arguments:
            css_data_dict {Mapping[str,str]} -- Dictionary ordered by the keys

        Returns:
            {list} -- List where every element contains a dictionary to create a Cross Section.
        """
        if not css_data_dict or not isinstance(css_data_dict, dict):
            return []

        number_of_css = len(css_data_dict[next(iter(css_data_dict))])
        css_dict_keys = css_data_dict.keys()
        css_dict_values = css_data_dict.values()
        css_data_list = [
            dict(zip(css_dict_keys, [value[idx]
            for value in css_dict_values if idx < len(value)]))
            for idx in range(number_of_css)]
        return css_data_list


class CrossSection:
    """
    Use this class to derive cross-sections from fm_data (2D model results). See docs how to acquire fm_data and how
    to prepare a proper 2D model.
    """
    __cs_parameter_css_points = 'number_of_css_points'
    __cs_parameter_transitionheight_sd = 'transitionheight_sd'
    __cs_parameter_velocity_threshold = 'velocity_threshold'
    __cs_parameter_relative_threshold = 'relative_threshold'
    __cs_parameter_min_depth_storage = 'min_depth_storage'
    __cs_parameter_plassen_timesteps = 'plassen_timesteps'
    __cs_parameter_storagemethod_wli = 'storagemethod_wli'
    __cs_parameter_bedlevelcriterium = 'bedlevelcriterium'
    __cs_parameter_SDstorage = 'SDstorage'
    __cs_parameter_Frictionweighing = 'Frictionweighing'
    __cs_parameter_sectionsmethod = 'sectionsmethod'

    __mask_point = None
    __logger = None

    def __init__(
            self,
            InputParam_dict: dict,
            name: str, length: float, location: tuple,
            branchid="not defined", chainage=0):
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
        # Cross-section meta data
        self.name = name                # cross-section id
        self.length = length            # 'vaklengte'
        self.location = location        # (x,y)
        self.branch = branchid          # name of 1D branch for cross-section
        self.chainage = chainage        # offset from beginning of branch

        # Cross-section geometry
        self.z = np.array([])
        self.total_width = np.array([])
        self.flow_width = np.array([])
        self.alluvial_width = 0.
        self.nonalluvial_width = 0.
        self.alluvial_friction_table = np.array([])
        self.nonalluvial_friction_table = np.array([])
        self.roughness_sections = np.array([])

        # delta h corrections ("summerdike option")
        self.crest_level = 0
        # in cross-section def. WAQ2PROF did crest - some fixed value.
        #  how to do here?
        self.floodplain_base = 0.0
        # note" 'to avoid numerical osscilation'. might need minimal value.
        # fixed or variable? Test!
        self.transition_height = 0.5
        self.extra_flow_area = 0.0
        self.extra_total_volume = 0.0
        self.extra_area_percentage = list()
        self.extra_total_area = 0

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

        # PARAMETERS
        self.temp_param_skip_maps = 2
        self.parameters = InputParam_dict

    def get_mask_point(self):
        return self.__mask_point

    # Public functions
    def build_from_fm(self, fm_data):
        """
        Build 1D geometrical cross-section from FM data.

        :param fm_data: dict
        :return:
        """

        # Unpack FM data
        waterlevel = fm_data['waterlevel'].iloc[:, self.temp_param_skip_maps:]
        waterdepth = fm_data['waterdepth'].iloc[:, self.temp_param_skip_maps:]
        velocity = fm_data['velocity'].iloc[:, self.temp_param_skip_maps:]
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
        self._set_logger_message('Retrieving centre point values')
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
        self._set_logger_message('Identifying lakes')

        # plassen_mask needed for arrays in output
        plassen_mask, wet_not_plas_mask, plassen_depth_correction = self._identify_lakes(waterdepth)

        # Masks for wet and flow cells (stroomvoeringscriteria)
        self._set_logger_message('Seperating flow from storage')
        flow_mask = self._distinguish_flow_from_storage(waterdepth, velocity)

        # Calculate area and volume as function of waterlevel & waterdepth
        self._fm_wet_area = np.nansum(area_matrix[wet_not_plas_mask], axis=0)
        self._fm_flow_area = np.nansum(area_matrix[flow_mask], axis=0)

        # Correct waterdepth for plassen
        waterdepth = waterdepth + plassen_depth_correction
        waterdepth = waterdepth[waterdepth >= 0]

        # Compute 2D volume as sum of area times depth
        self._fm_total_volume = np.array(np.nansum(area_matrix[wet_not_plas_mask] * waterdepth[wet_not_plas_mask], axis=0))
        self._fm_flow_volume = np.array(np.nansum(area_matrix[flow_mask] * waterdepth[flow_mask], axis=0))

        # For roughness we will need the original z-levels, since geometry z will change below
        self._css_z_roughness = centre_level

        # Check for monotonicity (water levels should rise)
        mono_mask = self._check_monotonicity(centre_level, method=1)
        centre_level = centre_level[mono_mask]
        self._fm_total_volume = self._fm_total_volume[mono_mask]
        self._fm_flow_volume = self._fm_flow_volume[mono_mask]
        self._fm_wet_area = self._fm_wet_area[mono_mask]
        self._fm_flow_area = self._fm_flow_area[mono_mask]

        # Compute geometry above z0 - Water level dependent calculation
        self._set_logger_message('Computing cross-section from water levels')
        self._compute_css_above_z0(centre_level)

        # Compute geometry below z0 - Water level independent calculation
        self._set_logger_message('Computing cross-section from bed levels')
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

    def optimisation_function(self, opt_in):
        """
        Objective function used in optimising a delta-h correction for parameters:
            crest_level         : level at which the correction begins
            transition_height   : height over which volume is released
            extra_volume        : total extra volume


        :param opt_in: tuple
        :return:
        """
        (crest_level, extra_volume) = opt_in
        transition_height = self.parameters[self.__cs_parameter_transitionheight_sd]
        predicted_volume = self._css_total_volume+FE.get_extra_total_area(self._css_z, crest_level, transition_height)*extra_volume
        return FE.return_volume_error(predicted_volume, self._fm_total_volume)

    def calculate_correction(self, transition_height: float):
        """
        Function to determine delta-h correction (previously known as 'summerdike option'.
        Optimises values for transition height, crest levels and added volume.

        Updates variables:
        self._css_total_volume_corrected



        TODO: to avoid numerical oscillation, transition_height need minimal value. Fixed or variable? Test!
        :return:
        """

        # Set initial values for optimisation of parameters
        initial_error = self._css_total_volume - self._fm_total_volume
        initial_crest = self._css_z[np.nanargmin(initial_error)]
        initial_volume = np.abs(initial_error[-1])

        self._set_logger_message("Initial crest: {} m".format(initial_crest), level='debug')
        self._set_logger_message("Initial volume: {} m".format(initial_volume), level='debug')
        # Optimise attributes
        opt = so.minimize(self.optimisation_function,
                          (initial_crest, initial_volume),
                          method='Nelder-Mead',
                          tol=1e-6)

        # Unpack optimisation results
        crest_level = opt['x'][0]
        extra_volume = opt['x'][1]
        self._set_logger_message("final costs: {}".format(opt['fun']), level='debug')
        self._set_logger_message("Optimizer msg: {}".format(opt['message']), level='debug')
        self._set_logger_message("Optimized crest: {} m".format(crest_level), level='debug')
        self._set_logger_message("Optimized volume: {} m".format(extra_volume), level='debug')
        
        if transition_height is None:
            transition_height = 0.5            
        extra_area_percentage = FE.get_extra_total_area(self._css_z, crest_level, transition_height)

        # Write to attributes
        self._css_total_volume_corrected = self._css_total_volume+extra_area_percentage*extra_volume
        self.crest_level = crest_level
        self.transition_height = transition_height
        self.extra_total_volume = extra_volume
        self.extra_total_area = extra_volume / self.length
        self.extra_area_percentage = extra_area_percentage
        self._css_is_corrected = True

    def assign_roughness(self, fm_data):
        """
        This function builds a table of Chezy values as function of water level 
        The roughnes is divides into two sections on the assumption of 
        an alluvial (smooth) and nonalluvial (rough) part of the total
        cross-section. This division is made based on the final timestep.
        """
        chezy_fm = fm_data['chezy'].iloc[:, self.temp_param_skip_maps:]
        # Get chezy values at last timestep
        end_values = chezy_fm.T.iloc[-1]

        #end_values[end_values == 0] = np.nan

        ## Remove points that belong to ponds
        #plassen_indices = np.where(self.plassen_mask)

        #node_indices = fm_data['face_nodes'][plassen_indices]
        #node_indices = np.unique(node_indices.flatten())

        #for link_index, chezy_value in enumerate(end_values):
        #    nodes = fm_data['edge_nodes'][link_index]

        #    if nodes[0] in node_indices and nodes[1] in node_indices:
        #        # exclude this value from end_values
        #        end_values[link_index] = np.nan

        ##end_values = end_values[~np.isnan(end_values)]

        # Find split point (chezy value) by variance minimisation
        variance_list = list()
        split_candidates = np.arange(min(end_values), max(end_values), 1)
        for split in split_candidates:
            variance_list.append(np.max([np.var(end_values[end_values>split]), np.var(end_values[end_values<split])]))

        splitpoint = split_candidates[np.nanargmin(variance_list)]

        # Assign datapoints to a cross-section
        C_sections = list()
        C_sections.append(end_values[end_values > splitpoint].index)
        C_sections.append(end_values[end_values < splitpoint].index)

        self.roughness_sections = C_sections
        # Find roughness tables for each section
        chezy = chezy_fm.values.T
        chezy[chezy==0] = np.nan
        self.alluvial_friction_table = [self._css_z_roughness, chezy_fm.T[C_sections[0]].mean(axis=1)]
        self.nonalluvial_friction_table = [self._css_z_roughness, chezy_fm.T[C_sections[1]].mean(axis=1)]

        # Dirty fix
        # Set width of sections
        C_sections_edge = list()
        C_sections_edge.append([i for i, x in enumerate(end_values > splitpoint) if x])
        C_sections_edge.append([i for i, x in enumerate(end_values < splitpoint) if x])

        self.alluvial_width = self._calc_roughness_width(splitpoint, C_sections_edge[0], fm_data)
        self.nonalluvial_width = self._calc_roughness_width(splitpoint, C_sections_edge[1], fm_data)

        #floodplain base level (temporary here)

        self.floodplain_base = self._calc_base_level(splitpoint, C_sections_edge[1], fm_data)

        # for visualisation later
        self.alluvial_edge_indices = C_sections_edge[0]
        self.nonalluvial_edge_indices = C_sections_edge[1]

    def reduce_points(self, n, method='visvalingam_whyatt', verbose=True):
        """
        Reduces the number of points from _css attributes to a preset maximum.

        Implemented vertex reduction methods:

        'visvalingam_whyatt'
        Based on Visvalingam, M and Whyatt J D (1993), "Line Generalisation by Repeated Elimination of Points",
        Cartographic J., 30 (1), 46 - 51

        :param n: int
        :param method: str
        :param verbose: boolean
        :return:
        """

        n_before_reduction = len(self._css_total_width)

        points = np.array([[self._css_z[i], self._css_total_width[i]] for i in range(n_before_reduction)])
        reduced_index = n_before_reduction - 1 # default is the same value as it came
        if method.lower() == 'visvalingam_whyatt':
            try:
                simplifier = PS.VWSimplifier(points)
                reduced_index = simplifier.from_number_index(n)
            except Exception as e:
                print('Exception thrown while using polysimplify: {}'.format(str(e)))

        # Write to attributes       
        self.z = self._css_z[reduced_index]
        self.total_width = self._css_total_width[reduced_index]
        self.flow_width = self._css_flow_width[reduced_index]

        self._set_logger_message("Cross-section reduced from {} to {} points".format(n_before_reduction, len(self.total_width)))

        self._css_is_reduced = True

    def set_logger(self, logger):
        """ should be given a logger object (python standard library) """
        assert isinstance(logger, logging.Logger), 'logger should be instance of logging.Logger class'
        
        self.__logger = logger

    # Private Functions

    def _calc_roughness_width(self, splitpoint, link_indices, fm_data):
        # get the 2 nodes for every alluvial edge
        section_nodes = fm_data['edge_nodes'][link_indices]
        section_nodes = np.unique(section_nodes.flatten())
         
        # get the faces for every node (start at index 1)
        faces = np.array([index for index, nodes in enumerate(fm_data['face_nodes_full']) for node in nodes if node in section_nodes])
        faces += 1

        (unique_faces, faces_count) = np.unique(faces, return_counts=True)

        # only faces that occur at least 3 times (2 edges, 3 nodes) belong to the section
        # this approach works for both square as well as triangular meshes
        section_faces = unique_faces[faces_count >= 3]

        mask = np.array([index in section_faces for index in range(1, fm_data['area_full'].size + 1)])
        section_width = np.sum(fm_data['area_full'][mask])/self.length

        return section_width

    def _calc_base_level(self, splitpoint, link_indices, fm_data):
        # get the 2 nodes for every alluvial edge
        section_nodes = fm_data['edge_nodes'][link_indices]
        section_nodes = np.unique(section_nodes.flatten())
         
        # get the faces for every node (start at index 1)
        faces = np.array([index for index, nodes in enumerate(fm_data['face_nodes']) for node in nodes if node in section_nodes])
        faces += 1

        (unique_faces, faces_count) = np.unique(faces, return_counts=True)

        # only faces that occur at least 4 times (2 edges, 3 nodes) belong to the section
        section_faces = unique_faces[faces_count >= 3]

        mask = np.array([index in section_faces for index in range(1, fm_data['area_full'].size + 1)])
        bedlevels = fm_data['bedlevel_full'][mask]

        return np.average(bedlevels)

    def _calc_chezy(self, depth, manning):
        return depth**(1/float(6)) / manning

    def _identify_lakes(self, waterdepth):
        """
        Find cells that do not have rising water depths for a period of time (__cs_parameter_plassen_timesteps)

        :param waterdepth: waterdepths from fm_data
        
        :return plassen_mask: mask of all cells that contain a lake
        :return wet_not_plas_mask: mask of all cells that are wet, but not a lake
        :return plassen_depth_correction: the depth of lake at the start of a computation
        """
        # check for non-rising waterlevels
        waterdepth_diff = np.diff(waterdepth, n=1, axis=-1)

        # this creates a series
        # for waal, 2 steps = 32 hours
        plassen_mask = (waterdepth.T.iloc[self.parameters[self.__cs_parameter_plassen_timesteps]] > 0) & \
                       (np.abs(waterdepth.T.iloc[self.parameters[self.__cs_parameter_plassen_timesteps]] - waterdepth.T.iloc[0]) <= 0.01)

        self.plassen_mask = plassen_mask

        plassen_mask_time = np.zeros((len(waterdepth.T), len(plassen_mask)), dtype=bool)
        plassen_mask_time[0,:] = plassen_mask

        plassen_depth_correction = np.zeros(waterdepth.shape, dtype=float)

        for i, depths in enumerate(waterdepth):
            plassen_depth_correction[plassen_mask, i] = -waterdepth.T.iloc[0][plassen_mask]

        # walk through dataframe in time, for each timestep check when to unmask a plassen cell
        for i, diff in enumerate(waterdepth_diff.T):
            final_mask = reduce(np.logical_and, [(diff <= 0.001), (plassen_mask_time[i] == True)])

            plassen_mask_time[i + 1,:] = final_mask
                
        plassen_mask_time = pd.DataFrame(plassen_mask_time).T

        # find all wet cells
        wet_mask = waterdepth > 0

        # correct wet cells for plassen
        wet_not_plas_mask = reduce(np.logical_and, [(wet_mask == True), (plassen_mask_time == False)])
        #print (wet_not_plas_mask)
        #pmtmask = plassen_mask_time == False
        #print (wet_mask & pmtmask)
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
            self._css_flow_width[-i] = np.min([self._css_flow_width[-i], self._css_flow_width[-i+1]])

        # fix error when flow_width > total_width (due to floating points)
        self._css_flow_width[self._css_flow_width > self._css_total_width] = self._css_total_width[self._css_flow_width > self._css_total_width]

    def _distinguish_flow_from_storage(self, waterdepth, velocity):
        """
        Defines mask which is False when flow is below a certain threshold. 
        (Stroomvoeringscriteriumfunctie)

        Arguments:
            waterdepth: pandas Dataframe with cell id for index and time in columns
            velocity: pandas Dataframe with cell id for index and time in columns

        Returns:
            flow_mask: pandas Dataframe with cell id for index and time in columns. 
                       True for flow, False for storage
        """
        flow_mask = (waterdepth > 0) & \
                    (velocity > self.parameters[self.__cs_parameter_velocity_threshold]) & \
                    (velocity > self.parameters[self.__cs_parameter_relative_threshold]*np.mean(velocity))

        # shallow flows should not be used for identifying storage (cells with small velocities are uncorrectly seen as storage)
        waterdepth_correction = (waterdepth > 0) & (waterdepth < self.parameters[self.__cs_parameter_min_depth_storage])

        # combine flow and depth mask to avoid assigning shallow flows to storage
        #flow_mask = reduce(np.logical_or, [(flow_mask == True), (waterdepth_correction == True)])
        flow_mask = flow_mask | waterdepth_correction

        return flow_mask
        
    def _extend_css_below_z0(self, centre_level, centre_depth, bedlevel_matrix, area_matrix, plassen_mask):
        """
        Extends the cross-sectional information below the water level at the first timestep,
        under assumption that the polygon formed by the water level and the bed level is convex. 
        It works by walking down to the bed level at the center point in 'virtual water level steps'. 
        At each step, we sum the area of cells width bedlevels which would be submerged at that 
        virtual water level. 
        """
        level_z0 = centre_level[0]
        bdata = bedlevel_matrix[~plassen_mask]
        bmask = bdata < level_z0
        
        bedlevels_below_z0 = bdata[bmask]
        lowest_level_below_z0 = np.nanmin(np.unique(bedlevels_below_z0.values.T[-1]))
        lowest_level_below_z0 = centre_level[0] - centre_depth[0]

        for unique_level_below_z0 in reversed(np.linspace(lowest_level_below_z0, level_z0, 10)):

            # count area
            areas = area_matrix[bedlevels_below_z0 <= unique_level_below_z0]
            area_at_unique_level = np.nansum(areas.values.T[-1])
            
            self._css_z = np.insert(self._css_z, 0, unique_level_below_z0)
            self._css_flow_width = np.insert(self._css_flow_width, 0, area_at_unique_level/self.length)
            self._css_total_width = np.insert(self._css_total_width, 0, area_at_unique_level/self.length)
            self._fm_wet_area = np.insert(self._fm_wet_area, 0, area_at_unique_level)
            self._fm_flow_area = np.insert(self._fm_flow_area, 0, area_at_unique_level) 
            self._fm_flow_volume = np.insert(self._fm_flow_volume, 0, np.nan) 
            self._fm_total_volume = np.insert(self._fm_total_volume, 0, np.nan) 

    def _set_logger_message(self, err_mssg: str, level='info'):
        """Sets message to logger if this is set.

        Arguments:
            err_mssg {str} -- Error message to send to logger.
        """
        if not self.__logger:
            return
        
        if level.lower() not in ['info', 'debug', 'warning', 'error', 'critical']:
            self.__logger.error("{} is not valid logging level.".format(level.lower()))

        if level.lower()=='info':
            self.__logger.info(err_mssg)
        elif level.lower()=='debug':
            self.__logger.debug(err_mssg)
        elif level.lower()=='warning':
            self.__logger.warning(err_mssg)
        elif level.lower()=='error':
            self.__logger.error(err_mssg)
        elif level.lower()=='critical':
            self.__logger.critical(err_mssg)
        
    @staticmethod
    def _check_monotonicity(arr, method=2):
        """
        for given input array, create mask such that when applied to the array, all values are monotonically rising

        method 1: remove values were z is falling from array
        method 2: sort array such that z is always rising (default)

        Arguments:
            arr: 1d numpy array

        return:
            mask such that arr[mask] is monotonically rising
        """
        if method == 1:
            mask = np.array([True])
            for i in range(1,len(arr)):
                # Last index that had rising value
                j = np.argwhere(mask==True)[-1][0]
                if arr[i] > arr[j]:
                    mask = np.append(mask, True)
                else:
                    mask = np.append(mask, False)

            return mask
        elif method==2:
            return np.argsort(arr)

class ElapsedFormatter():
    __new_iteration = True
    def __init__(self):
        self.start_time = time()
        self.number_of_iterations = 1
        self.current_iteration = 0

    def format(self, record):
        if self.__new_iteration:
            return self.__format_header(record)
        else:
            return self.__format_message(record)

    def __format_header(self, record):
        self.__new_iteration = False
        elapsed_seconds = record.created - self.start_time
        return "{now} {level:>7} :: {progress:4.0f}% :: {message} ({file})".format(
                                                        now=datetime.now().strftime("%Y-%m-%d %H:%M"), 
                                                        level=record.levelname, 
                                                        elapsed=elapsed_seconds, 
                                                        message=record.getMessage(), 
                                                        file=record.filename,
                                                        progress=100*self.current_iteration/self.number_of_iterations)

    def __format_message(self, record):
        elapsed_seconds = record.created - self.start_time
        return "{now} {level:>7} :: {progress:4.0f}% ::   > T+ {elapsed:.2f}s {message} ({file})".format(
                                                        now=datetime.now().strftime("%Y-%m-%d %H:%M"), 
                                                        level=record.levelname, 
                                                        elapsed=elapsed_seconds, 
                                                        message=record.getMessage(), 
                                                        file=record.filename,
                                                        progress=100*self.current_iteration/self.number_of_iterations)

    def __reset(self):
        self.start_time = time()

    def start_new_iteration(self):
        self.current_iteration += 1
        self.next_step()

    def next_step(self):
        self.__new_iteration = True
        self.__reset()

    def set_number_of_iterations(self, n):
        assert n>0, 'Total number of iterations should be higher than zero'
        self.number_of_iterations=n