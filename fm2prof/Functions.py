#! /usr/bin/env python
"""
This module contains functions used for the emulation/reduction of 2D models to 1D models for Delft3D FM (D-Hydro).


Dependencies
------------------
Packages, between parenthesis known working version.

netCDF4 (1.2.1)
numpy (1.10.4)
pandas (0.17.1)
sklearn (0.15.2)

Contact: K.D. Berends (koen.berends@deltares.nl, k.d.berends@utwente.nl)
"""

# region // imports
import netCDF4
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

import os
import matplotlib.font_manager as font_manager
# endregion

# region // Module information
__author__ = "Koen Berends"
__copyright__ = "Copyright 2016, University of Twente & Deltares"
__credits__ = ["Koen Berends"]
__license__ = "no license (restricted)"
__version__ = "$Revision$"
__maintainer__ = "Koen Berends"
__email__ = "koen.berends@deltares.nl"
__status__ = "Prototype"
# endregion


# region // public functions
def read_fm2prof_input(res_file, css_file):
    """
    Reads input files for 'FM2PROF'. See documentation for file format descriptions.
    
    Data is saved in three major structures:
        time_independent_data: holds bathymetry information
        time_dependent_data: waterlevels, roughnesses and velocities
        edge_data: the nodes that relate to edges

    :param res_file: str, path to FlowFM map netcfd file (*_map.nc)
    :param css_file: str, path to cross-section definition file
    :return:
    """

    # Read FM map file
    (time_independent_data, edge_data, node_coordinates, time_dependent_data) = _read_fm_model(res_file)

    # Load locations and names of cross-sections
    cssdata = _read_css_xyz(css_file)

    # Create a class identifier to map points to cross-sections
    neigh = _get_class_tree(cssdata['xy'], cssdata['id'])

    # Expand time-independent dataset with cross-section names
    time_independent_data['sclass'] = neigh.predict(np.array([time_independent_data['x'], time_independent_data['y']]).T)

    # Assign cross-section names to edge coordinates as well
    edge_data['coordinates']['sclass'] = neigh.predict(np.array([edge_data['coordinates']['x'], edge_data['coordinates']['y']]).T)

    return time_dependent_data, time_independent_data, edge_data, node_coordinates, cssdata

def get_fm2d_data_for_css(classname, dti, edge_data, dtd):
    """
    create a dictionary that holds all the 2D data for the cross-section with name 'classname'
    """
    x = dti['x'][dti['sclass'] == classname]
    y = dti['y'][dti['sclass'] == classname]
    area = dti['area'][dti['sclass'] == classname]

    waterdepth = dtd['waterdepth'][dti['sclass'] == classname]
    waterlevel = dtd['waterlevel'][dti['sclass'] == classname]
    vx = dtd['velocity_x'][dti['sclass'] == classname]
    vy = dtd['velocity_y'][dti['sclass'] == classname]

    # find all chezy values for this cross section, note that edge coordinates are used
    chezy = dtd['chezy_edge'][edge_data['coordinates']['sclass'] == classname]
    edge_nodes = edge_data['edge_nodes'][edge_data['coordinates']['sclass'] == classname]

    velocity_edge = dtd['velocity_edge'][edge_data['coordinates']['sclass'] == classname]
    
    # retrieve the full set for face_nodes and area, needed for the roughness calculation
    face_nodes = edge_data['face_nodes'][dti['sclass'] == classname]
    face_nodes_full = edge_data['face_nodes']
    area_full = dti['area']
    bedlevel_full = dti['bedlevel']
    bedlevel = dti['bedlevel'][dti['sclass'] == classname]

    velocity = (vx**2+vy**2)**0.5
    waterlevel[waterdepth == 0] = np.nan

    return_dict = {
        'x': x, 
        'y': y, 
        'bedlevel': bedlevel, 
        'bedlevel_full': bedlevel_full, 
        'waterdepth': waterdepth, 
        'waterlevel': waterlevel, 
        'velocity': velocity, 
        'area': area, 
        'chezy': chezy, 
        'edge_nodes': edge_nodes, 
        'face_nodes': face_nodes, 
        'face_nodes_full': face_nodes_full, 
        'area_full': area_full}

    return return_dict

def mirror(array, reverse_sign=False):
    """
    Mirrors array

    :param array:
    :param reverse_sign:
    :return:
    """
    if reverse_sign:
        return np.append(np.flipud(array)*-1, array)
    else:
        return np.append(np.flipud(array), array)

def get_centre_values(location, x, y, waterdepth, waterlevel):
    """
    Find output point closest to x,y location, output depth and water level as nd arrays
    
    """
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array([x, y]).T)

    # conversion to 2d array, as 1d arrays are deprecated for kneighbors
    location_array = np.array(location).reshape(1, -1)
    (distance, index) = nn.kneighbors(location_array)

    # retrieve cell characteristic waterdepth
    centre_depth = waterdepth.iloc[index[0]]
    centre_level = waterlevel.iloc[index[0]]

    return centre_depth.values[0], centre_level.values[0]

def get_extra_total_area(waterlevel, crest_level, transition_height, hysteresis=False):
    """
    releases extra area dependent on waterlevel using a logistic (sigmoid) function
    """
    return 1/(1+np.e**(np.log(0.00001)/(transition_height)*(waterlevel-(crest_level+0.5*transition_height))))

def return_volume_error(predicted, measured, gof='rmse'):
    error = np.array(predicted - measured)/np.maximum(np.array(measured), np.ones(len(measured)))
    return np.sum(error**2)

def interpolate_roughness(cross_section_list):
    try:
        # get alluvial roughness tables from all cross sections
        alluvial_levels = [css.alluvial_friction_table[0] for css in cross_section_list]
        alluvial_chezy = [css.alluvial_friction_table[1] for css in cross_section_list]

        # same for nonalluvial
        nonalluvial_levels = [css.nonalluvial_friction_table[0] for css in cross_section_list]
        nonalluvial_chezy = [css.nonalluvial_friction_table[1] for css in cross_section_list]
    except IndexError:
        return None

    # construct ranges
    alluvial_range = _construct_range(alluvial_levels, 0.01)
    nonalluvial_range = _construct_range(nonalluvial_levels, 0.01)

    # interpolate all values in every cross section using the constructed ranges
    for css in cross_section_list:
        _interpolate_roughness_css(css, alluvial_range, nonalluvial_range)

# endregion

# region // protected functions
def _read_fm_model(file_path):
    """input: FM2D map file"""

    # Open results file for reading
    res_fid = netCDF4.Dataset(file_path, 'r')

    # Time-invariant variables from FM 2D
    df = pd.DataFrame(columns=['x'], data=np.array(res_fid.variables['mesh2d_face_x']))
    df['y'] = np.array(res_fid.variables['mesh2d_face_y'])
    df['area'] = np.array(res_fid.variables['mesh2d_flowelem_ba'])
    df['bedlevel'] = np.array(res_fid.variables['mesh2d_flowelem_bl'])

    # Time-variant variables
    time_dependent = {
                      'waterdepth': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_waterdepth']).T, columns=res_fid.variables['time']),
                      'waterlevel': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_s1']).T, columns=res_fid.variables['time']),
                      'chezy_mean': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_czs']).T, columns=res_fid.variables['time']),
                      'chezy_edge': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_cftrt']).T, columns=res_fid.variables['time']),
                      'velocity_x': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_ucx']).T, columns=res_fid.variables['time']),
                      'velocity_y': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_ucy']).T, columns=res_fid.variables['time']),
                      'velocity_edge': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_u1']).T, columns=res_fid.variables['time'])
                      }

    df_edge = pd.DataFrame(columns=['x'], data=np.array(res_fid.variables['mesh2d_edge_x']))
    df_edge['y'] = np.array(res_fid.variables['mesh2d_edge_y'])

    edge_data = {
                'coordinates': df_edge,
                'edge_nodes': np.array(res_fid.variables['mesh2d_edge_nodes']),
                'face_nodes': np.array(res_fid.variables['mesh2d_face_nodes'])
                }

    df_node = pd.DataFrame(columns=['x'], data=np.array(res_fid.variables['mesh2d_node_x']))
    df_node['y'] = np.array(res_fid.variables['mesh2d_node_y'])
    
    return df, edge_data, df_node, time_dependent

def _read_css_xyz(file_path : str, delimiter = ','):
    
    if not file_path or not os.path.exists(file_path):
        raise IOError('No file path for Cross Section location file was given, or could not be found at {}'.format(file_path))
    
    with open(file_path, 'r') as fid:
        input_data = dict(xy=list(), id=list(), branchid=list(), length=list(), chainage=list())
        for line in fid:
            try:
                (x, y, branchid, length, chainage) = line.split(delimiter)
            except ValueError:
                # revert to legacy format
                (x, y, length) = line.split(delimiter)
                branchid = 'not defined'
                chainage = 0

            input_data['xy'].append((float(x), float(y)))
            input_data['id'].append(branchid + '_' + str(round(float(chainage))))
            input_data['length'].append(float(length))
            input_data['branchid'].append(branchid)
            input_data['chainage'].append(float(chainage))
        return input_data

def _get_class_tree(xy, c):
    X = xy
    y = c
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X, y)
    return neigh

def _construct_range(xp_list, step):
    xp_flat = [item for item in xp_list]

    min_xp = np.min(xp_flat) - 0.3
    max_xp = np.max(xp_flat) + 0.3

    return np.arange(min_xp, max_xp, step)

def _interpolate_roughness_css(cross_section, alluvial_range, nonalluvial_range):
    # change nan's to zeros
    chezy_alluvial = np.nan_to_num(cross_section.alluvial_friction_table[1])
    chezy_nonalluvial = np.nan_to_num(cross_section.nonalluvial_friction_table[1])
    chezy_nonalluvial = np.nan_to_num(cross_section.nonalluvial_friction_table[1])

    waterlevel_alluvial = cross_section.alluvial_friction_table[0]
    waterlevel_nonalluvial = cross_section.nonalluvial_friction_table[0]

    # remove trailing zeros
    chezy_alluvial_trimmed = np.trim_zeros(chezy_alluvial)
    chezy_nonalluvial_trimmed = np.trim_zeros(chezy_nonalluvial)

    alluvial_nonzero_mask = np.nonzero(chezy_alluvial)[0]
    nonalluvial_nonzero_mask = np.nonzero(chezy_nonalluvial)[0]

    # only interpolate and assign if nonzero elements exist in the chezy table
    if np.sum(alluvial_nonzero_mask) > 0:
        waterlevel_alluvial_trimmed = waterlevel_alluvial[alluvial_nonzero_mask[0]:alluvial_nonzero_mask[-1] + 1]
        alluvial_interp = np.interp(alluvial_range, waterlevel_alluvial_trimmed, chezy_alluvial_trimmed)

        # assign
        cross_section.alluvial_friction_table[0] = alluvial_range
        cross_section.alluvial_friction_table[1] = pd.Series(data=alluvial_interp)

    if np.sum(nonalluvial_nonzero_mask) > 0:
        waterlevel_nonalluvial_trimmed = waterlevel_nonalluvial[nonalluvial_nonzero_mask[0]:nonalluvial_nonzero_mask[-1] + 1]
        nonalluvial_interp = np.interp(nonalluvial_range, waterlevel_nonalluvial_trimmed, chezy_nonalluvial_trimmed)

        # assign
        cross_section.nonalluvial_friction_table[0] = nonalluvial_range
        cross_section.nonalluvial_friction_table[1] = pd.Series(data=nonalluvial_interp)


# endregion
