﻿#! /usr/bin/env python
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cmx
from functools import reduce
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as so
from time import time
import seaborn as sns

from fm2prof import common
from fm2prof import Functions as FE
from typing import Mapping, Sequence
from .lib import polysimplify as PS

import os
import matplotlib.font_manager as font_manager

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
    def __init__(self, arg_list : list):
        if not arg_list:
            raise Exception('FM model data was not read correctly.')
        if len(arg_list) != 5:
            raise Exception ('Fm model data expects 5 arguments but only {} were given'.format(len(arg_list)))
        
        (self.time_dependent_data, self.time_independent_data, self.edge_data, self.node_coordinates, css_data_dictionary) = arg_list
        self.css_data_list = self.get_ordered_css_list(css_data_dictionary)

    @staticmethod
    def get_ordered_css_list(css_data_dict : Mapping[str,str]):
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
        css_data_list = [dict(zip(css_dict_keys, [value[idx] for value in css_dict_values if idx < len(value)])) for idx in range(number_of_css)]
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

    def __init__(self, InputParam_dict, name : str, length : float, location : tuple, branchid = "not defined", chainage=0):
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
        self.floodplain_base = 0.0 # in cross-section def. WAQ2PROF did crest - some fixed value. how to do here?
        self.transition_height = 0.5 # note" 'to avoid numerical osscilation'. might need minimal value. fixed or variable? Test!
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

        # DEFAULTS
        self.color_palette = 'Tableau20'
        self._set_plotstyle(style=3)

        # PARAMETERS
        self.temp_param_skip_maps = 2
        self.parameters = InputParam_dict

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

        # Convert area to a matrix for matrix operations (much more efficient than for-loops)
        area_matrix = pd.DataFrame(index=area.index)
        for t in waterdepth:
            area_matrix[t] = area

        bedlevel_matrix = pd.DataFrame(index=bedlevel.index)
        for t in waterdepth:
            bedlevel_matrix[t] = bedlevel

        # Retrieve the water-depth & water level nearest to the cross-section location
        (centre_depth, centre_level) = FE.get_centre_values(self.location, fm_data['x'], fm_data['y'], waterdepth, waterlevel)

        # apply rolling average over the velocities to smooth out extreme values
        velocity = velocity.rolling(window=10, min_periods=1, center=True, axis=1).mean()

        # Identify river lakes (plassen)
        plassen_mask, wet_not_plas_mask, plassen_depth_correction = self.__identify_lakes(waterdepth)  

        # Masks for wet and flow cells (stroomvoeringscriteria)
        flow_mask = self.__distinguish_flow_from_storage(waterdepth, velocity)
               
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
        self.__compute_css_above_z0(centre_level)
        
        # Compute geometry below z0 - Water level independent calculation
        self.__extend_css_below_z0(centre_level, centre_depth, bedlevel_matrix, area_matrix, plassen_mask)

        # Compute 1D volume as integral of width with respect to z times length
        self._css_total_volume = np.append([0], np.cumsum(self._css_total_width[1:]*np.diff(self._css_z)*self.length))
        self._css_flow_volume = np.append([0], np.cumsum(self._css_flow_width[1:]*np.diff(self._css_z)*self.length))
        
        # If sd correction is run, these attributed will be updated. 
        self._css_total_volume_corrected = self._css_total_volume
        self._css_flow_volume_corrected = self._css_flow_volume

        # convert to float64 array for uniformity (apparently entries can be float32)
        self._css_z = np.array(self._css_z, dtype=np.dtype('float64')) 
        
    def __distinguish_flow_from_storage(self, waterdepth, velocity):
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
        
    def __extend_css_below_z0(self, centre_level, centre_depth, bedlevel_matrix, area_matrix, plassen_mask):
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

    def calculate_correction(self, transition_height:float):
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

        # Optimise attributes
        opt = so.minimize(self.optimisation_function,
                          (initial_crest, initial_volume),
                          method='Nelder-Mead',
                          tol=1e-6)

        # Unpack optimisation results
        crest_level = opt['x'][0]
        extra_volume = opt['x'][1]

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

    def _plot_roughness(self, fm_data, b_plot_nonalluvial=True):
        #sns.set(style='ticks')
        #sns.set(font_scale=1.5)

        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111)

        colors = sns.color_palette()

        #for i in fm_data['chezy'].T:
        #    if i in self.roughness_sections[0]:
        #        ax.plot(self._css_z, fm_data['chezy'].T[i].values, '.', color=[0.6, 0.6, 0.6])
        #    elif i in self.roughness_sections[1]:
        #        ax.plot(self._css_z, fm_data['chezy'].T[i].values, '^', markeredgecolor=[0.6, 0.6, 0.6], color=[0.6, 0.6, 0.6])

        depth_main = np.append(np.arange(0.1, self.alluvial_friction_table[0][0] - np.min(self._css_z), 0.1), self.alluvial_friction_table[0] - np.min(self._css_z))
        chezy_main = self._calc_chezy(depth_main, 0.03)

        depth_flood = np.append(np.arange(0.1, self.alluvial_friction_table[0][0] - 0, 0.1), self.nonalluvial_friction_table[0] - 0)
        chezy_flood = self._calc_chezy(depth_flood, 0.07)

        #depth_flood_1 = depth_flood - 0.5
        #chezy_flood_1 = self._calc_chezy(depth_flood_1, 0.07)

        ax.plot(depth_main - 2.5, chezy_main, '-', color=colors[0], linewidth=2, alpha=0.4, label='Analytic calculation')

        if b_plot_nonalluvial:
            ax.plot(depth_flood-0.5, chezy_flood, '-', color=colors[0], linewidth=2, alpha=0.4)

        #ax.plot(depth_flood_1 + 0.5, chezy_flood_1, '-', color=colors[0], linewidth=2, alpha=0.4)

        ax.plot(self.alluvial_friction_table[0], self.alluvial_friction_table[1], '--', dashes=(6, 12), color=colors[1], linewidth=3, label='Alluvial roughness')

        if b_plot_nonalluvial:
            ax.plot(self.nonalluvial_friction_table[0], self.nonalluvial_friction_table[1], '--', dashes=(3, 3), color=colors[2], linewidth=3, label='Nonalluvial roughness')

        ax.set_xlabel('Waterlevel [m]')
        ax.set_ylabel('Chezy roughness coefficient $[\mathrm{m^{0.5}/s}]$')
        #ax.set_title('Chezy roughness for cross-section')
        ax.legend(loc=2)

        ax.set(xlim=(np.min(self._css_z), np.max(self.alluvial_friction_table[0]) + 0.2))
        ax.set(ylim=(0, np.max(self.alluvial_friction_table[1]) + 5))

        #plt.gca().set_aspect('equal', adjustable='box')
        sns.despine()
        return fig
    
    def _calc_chezy(self, depth, manning):
        return depth**(1/float(6)) / manning

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

        #points = np.array([[self._css_z[i], self._css_total_width[i], self._css_flow_width[i]] for i in range(n_before_reduction)])
        points = np.array([[self._css_z[i], self._css_total_width[i]] for i in range(n_before_reduction)])
        reduced_index = n_before_reduction - 1 # default is the same value as it came
        if method.lower() == 'visvalingam_whyatt':
            try:
                simplifier = PS.VWSimplifier(points)
                reduced_index = simplifier.from_number_index(n)
            except Exception as e:
                print('Exception thrown while using polysimplify: {}'.format(str(e)))

        # Write to attributes
        #self.z = np.array(self._css_z)
        #self.total_width = self._css_total_width
        #self.flow_width = self._css_flow_width
        
        self.z = self._css_z[reduced_index]
        self.total_width = self._css_total_width[reduced_index]
        self.flow_width = self._css_flow_width[reduced_index]

        if verbose:
            print("Reduced from %s to %s points in %03f seconds" % (n_before_reduction, n, end_time-start_time))

        self._css_is_reduced = True

    @staticmethod
    def _set_plotstyle(style=2):
        """
        Set plotting style. 

        Keyword argument:

        style : int, 1 = 'ggplot' style
                     2 = 'seaborn whitegrid style'
        """
        if style == 1:
            # 'GGPLOT' style
            plt.style.use('ggplot')
            font = {'family': 'sans-serif',
                    'size': 18}
            mpl.rc('font', **font)
        elif style == 2:
            # Style more suited for publication
            sns.set(font_scale=2)
            sns.set_style("whitegrid", {""
                "font.family": "serif",
                "font.serif": ["Times", "Palatino", "serif"]  # ,"font.size": 22
            })

            sns.set_context("paper", rc={"axes.titlesize":18,"axes.labelsize":15}) # "font.size":18,
            sns.set(font='serif')
        elif style == 3:
            # Style more suited for publication

            # Add custom font
            path = os.path.join(os.path.dirname(__file__))
            font_dirs = [path, ]
            font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
            font_list = font_manager.createFontList(font_files)
            font_manager.fontManager.ttflist.extend(font_list)

            #vitamin_c_palette = ["#004358", "#FD7400", "#1F8A70", "#FFE11A", "#BEDB39"]

            #sns.set_context("paper", rc={"text.usetex": True, "lines.linewidth": 4})
            #sns.set("paper", font="Linux Libertine O", font_scale=2.5, style='whitegrid', palette=vitamin_c_palette)

            # Set color palette

            #sns.set_palette(vitamin_c_palette)
            
    def _plot_2d_model_grid(self, ax_grid, fm_data, node_coordinates, color, type='bedlevel'):
        colors = sns.color_palette()

        # plot nodes as points
        if type == 'bedlevel':
            scatter = ax_grid.scatter(fm_data['x'], fm_data['y'], c=list(fm_data['bedlevel']), cmap='viridis', s=30)
            #ax_grid.set_xlabel('x-coordinate [m]', fontsize=14)
            #ax_grid.set_ylabel('y-coordinate [m]', fontsize=14)
            cbar = plt.colorbar(scatter)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Bedlevel [m]', fontsize=14)

            # plot line sections between nodes
            for nodes in fm_data['edge_nodes']:
                nodes -= 1
                x_coordinates = np.array(node_coordinates['x'][nodes])
                y_coordinates = np.array(node_coordinates['y'][nodes])

                ax_grid.plot(x_coordinates, y_coordinates, color=color, alpha=1)
        elif type == 'links':
            roughness = np.array(fm_data['chezy'].T.iloc[-1])

            #remove nan values
            roughness = roughness[~np.isnan(roughness)]

            color_info = ax_grid.contourf([[0,0],[0,0]], np.sort(np.unique(roughness)), cmap='viridis')
            plt.cla()

            cmap = plt.get_cmap('viridis')
            cnorm = mcolors.Normalize(vmin=np.min(roughness), vmax=np.max(roughness))
            scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

            ax_grid.set_axis_bgcolor('w')
            ax_grid.grid(False)
            ax_grid.set_xticklabels([])
            ax_grid.set_yticklabels([])

            for index, nodes in enumerate(fm_data['edge_nodes']):
                nodes -= 1

                x_coordinates = np.array(node_coordinates['x'][nodes])
                y_coordinates = np.array(node_coordinates['y'][nodes])
                value = fm_data['chezy'].T.iloc[-1].iloc[index]
                colorVal = scalarMap.to_rgba(value)

                ax_grid.plot(x_coordinates, y_coordinates, color=colorVal, alpha=1)

            cbar = plt.colorbar(color_info, format='%d')
            cbar.ax.tick_params(labelsize=14)
            #cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
            cbar.set_label(r'$\mathrm{Ch\acute{e}zy}$ roughness coefficient $[\mathrm{m^{0.5}/s}]$', fontsize=14)
            #colorbar = mpl.colorbar.ColorbarBase(ax_grid, cmap=cmap, norm=cnorm, orientation='vertical')
            #colorbar.set_label('Ch\'{e}zy roughness coefficient $[\mathrm{m^{0.5}/s}]$')
        else:
            # plot line sections between nodes
            for nodes in fm_data['edge_nodes']:
                nodes -= 1
                x_coordinates = np.array(node_coordinates['x'][nodes])
                y_coordinates = np.array(node_coordinates['y'][nodes])

                ax_grid.plot(x_coordinates, y_coordinates, color=color, alpha=1)

        plt.gca().set_aspect('equal', adjustable='box')

    def _plot_roughness_sections(self, full_grid_ax, fm_data, node_coordinates):
        alluvial_nodes = fm_data['edge_nodes'][self.alluvial_edge_indices]
        nonalluvial_nodes = fm_data['edge_nodes'][self.nonalluvial_edge_indices]

        colors = sns.color_palette()

        for index, node_indices in enumerate(alluvial_nodes):
            x_coordinates = node_coordinates['x'][node_indices]
            y_coordinates = node_coordinates['y'][node_indices]

            if index == 0:
                full_grid_ax.plot(x_coordinates, y_coordinates, linewidth=1, color=colors[0])
            else:
                full_grid_ax.plot(x_coordinates, y_coordinates, linewidth=1, color=colors[0])

        for index, node_indices in enumerate(nonalluvial_nodes):
            x_coordinates = node_coordinates['x'][node_indices]
            y_coordinates = node_coordinates['y'][node_indices]

            if index == 0:
                full_grid_ax.plot(x_coordinates, y_coordinates, linewidth=1, color=colors[1])
            else:
                full_grid_ax.plot(x_coordinates, y_coordinates, linewidth=1, color=colors[1])

        patch_alluvial = mpatches.Patch(color=colors[0], label='Alluvial')
        patch_nonalluvial = mpatches.Patch(color=colors[1], label='Nonalluvial')

        full_grid_ax.legend(loc=2, handles=[patch_alluvial, patch_nonalluvial])

    def _plot_ponds(self, ax, fm_data, node_coordinates):
        colors = sns.color_palette()

        # plot line sections between nodes
        for nodes in fm_data['edge_nodes']:
            nodes -= 1
            x_coordinates = np.array(node_coordinates['x'][nodes])
            y_coordinates = np.array(node_coordinates['y'][nodes])

            ax.plot(x_coordinates, y_coordinates, color='k', alpha=1)

        plassen_x = fm_data['x'][self.plassen_mask]
        plassen_y = fm_data['y'][self.plassen_mask]

        ax.scatter(plassen_x, plassen_y, color=colors[1], s=30)
        plt.gca().set_aspect('equal', adjustable='box')
            
    def _plot_zw(self, z_prof=None, w_prof=None):
        """
        Plot z/width graph.

        :param ax: float, optional handle to pyplot axes object
        :return:
        """
        #sns.set(style='ticks')
        #sns.set(font_scale=1.5)

        fig = plt.figure(figsize=(10, 7))
        #gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[6, 1])

        #hLeft = fig.add_subplot(gs[0])
        ax = fig.add_subplot(111)

        #colors = common.get_color_palette()
        #if self._css_is_reduced:
        #    hLeft.plot(self.total_width, self.z,
        #               linestyle='-',
        #               marker='x',
        #               color=colors[0],
        #               label='Reduced')

        # plot original profile
        if w_prof != None:
            ax.plot(w_prof, z_prof, linestyle='-', alpha=0.4, linewidth=2, label='Original')
        
        if self._css_is_reduced:
            ax.plot(np.append([0], self.total_width * 0.5), np.append([self.z[0]], self.z), linewidth=2,
                   linestyle='--',
                   label='Generated')
            ax.set(ylim=(np.min(self.z) - 0.1, np.max(self.z) + 0.1))
        else:
            ax.plot(np.append([0], self._css_total_width * 0.5), np.append([self._css_z[0]], self._css_z), linewidth=2,
                   linestyle='--',
                   label='Generated')

            ax.set(ylim=(np.min(self._css_z) - 0.1, np.max(self._css_z) + 0.1))


        plt.ylabel('Waterlevel [m]')
        plt.xlabel('Width [m]')
        plt.legend(loc=2)
        sns.despine()
        return fig

        #hRight = fig.add_subplot(gs[1])
        #hRight.plot(self.extra_area_percentage, self._css_z, linewidth=2, color=colors[6])
        #hRight.text(0.5, self.crest_level,
        #            '%.0f m2\n(%.1f pct)' %
        #            (self.extra_total_volume/self.length, 100*self.extra_total_volume/self._css_total_volume[-1]),
        #            horizontalalignment='center')

        #hRight.set_yticks(hLeft.get_yticks())
        #hRight.set_ylim(hLeft.get_ylim())
        #hRight.set_xticks([0, 1])
        #hRight.set_title('Delta-h correction')
        #plt.setp(hRight.get_yticklabels(), visible=False)
        #plt.subplots_adjust(right=0.85)
        #hRight.set_axis_bgcolor(colors[1])
        #plt.tight_layout()

    def _plot_volume(self, ax=None, relative_error=True):
        """
        Plots (2x1) plot of waterlevel/volume and waterlevel/volume error

        :param ax:
        :param relative_error:
        :return:
        """

        #colors = common.get_color_palette(self.color_palette)

        #if not ax:
        fig = plt.figure(figsize=(14, 7))
        hError = fig.add_subplot(212)
        hVolume = fig.add_subplot(211)

        if relative_error:
            hError.plot(self._css_z, 100*np.array(self._css_total_volume - self._fm_total_volume)/np.array(self._fm_total_volume), linewidth=2)
            if self._css_is_corrected:
                hError.plot(self._css_z, 100*np.array(self._css_total_volume_corrected - self._fm_total_volume) /
                            np.array(self._fm_total_volume), '--', linewidth=2)

                hError.set_ylabel('Relative error [%]')
        else:
            hError.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            hError.plot(self._css_z, np.array(self._css_total_volume - self._fm_total_volume), linewidth=2)
            if self._css_is_corrected:
                hError.plot(self._css_z, np.array(self._css_total_volume_corrected - self._fm_total_volume), linewidth=2)

            hError.set_ylabel('Error of total volume [$\mathrm{m^3}$]')

        hError.set_xlabel('Water level [m]')

        #hVolume.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

        hVolume.plot(self._css_z, np.array(self._fm_total_volume)*1e-6, 'k', linewidth=2, alpha=0.5, label='2D model results')
        hVolume.plot(self._css_z, np.array(self._css_total_volume)*1e-6, '--', linewidth=2, label='1D result without correction')
        if self._css_is_corrected:
            hVolume.plot(self._css_z, np.array(self._css_total_volume_corrected)*1e-6, '--', linewidth=2, label='1D result with correction')

        plt.legend(loc=2)
        hVolume.set_ylabel('Total volume [million $\mathrm{m^3}$]')
        plt.setp(hVolume.get_xticklabels(), visible=False)
        return fig

    def _plot_fm(self, fm_data):
        fig = plt.figure(figsize=(20, 10))
        h3dPlot = fig.add_subplot(111, projection='3d')

        h3dPlot.scatter(list(fm_data['x']), list(fm_data['y']), list(fm_data['z']), c=fm_data['z'])

        h3dPlot.scatter(self.location[0], self.location[1], np.max(fm_data['z']), c='k', marker='^', s=40)
        plt.gca().invert_zaxis()
        plt.title('Delft FM points attributed to cross-section')
        return fig
    
    def _close_figure(self, figure):
        plt.close(figure)

    def __identify_lakes(self, waterdepth):
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

    def __compute_css_above_z0(self, centre_level):
        """
        'Area method': compute width from area
        
        """

        self._css_z = centre_level

        # Remove nan's from z
        self._css_z[np.isnan(self._css_z)] = np.nanmin(self._css_z)
        
        # Compute widths
        self._css_total_width = np.array(self._fm_wet_area)/self.length
        self._css_flow_width = np.array(self._fm_flow_area)/self.length

        # fix error when flow_width > total_width (due to floating points)
        self._css_flow_width[self._css_flow_width > self._css_total_width] = self._css_total_width[self._css_flow_width > self._css_total_width]

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

class Logger:
    def __init__(self, outputdir):
        self._outputdir = outputdir
        with open('{}/{}.log'.format(outputdir, 'fm2prof'), 'w') as f:
            pass

    
    def write(self, arg):
        with open('{}/{}.log'.format(self._outputdir, 'fm2prof'), 'a') as f:
            f.write("{}\n".format(arg))
        print(arg)