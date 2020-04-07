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
"""
Copyright (C) Stichting Deltares 2019. All rights reserved.

This file is part of the Fm2Prof.

The Fm2Prof is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

All names, logos, and references to "Deltares" are registered trademarks of
Stichting Deltares and remain full property of Stichting Deltares at all times.
All rights reserved.
"""
# region // imports
import netCDF4
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import csv

import os

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
def classify_roughness_sections_by_variance(data, variable):

    # Get chezy values at last timestep
    end_values = variable.T.iloc[-1].values
    key = 'section'
    # Find split point (chezy value) by variance minimisation
    variance_list = list()
    split_candidates = np.arange(min(end_values), max(end_values), 1)
    if len(split_candidates) < 2: # this means that all end values are very close together, so do not split
        data[key][:] = 'main'
    else:
        for split in split_candidates:
            variance_list.append(
                np.max(
                    [np.var(end_values[end_values > split]),
                     np.var(end_values[end_values <= split])]))

        splitpoint = split_candidates[np.nanargmin(variance_list)]

        # High chezy values are assigned to section number '1' (Main channel)
        data[key][end_values > splitpoint] = 'main'

        # Low chezy values are assigned to section number '2' (Flood plain) 
        data[key][end_values <= splitpoint] = 'floodplain1'
    return data


def classify_roughness_sections_by_polygon(sections, data, logger):
    """ assigns edges to a roughness section based on polygon data """
    logger.debug('....gathering points')
    points = [(data['x'][i], data['y'][i])
              for i in range(len(data['x']))]
    logger.debug('....classifying points')
    data['section'] = sections.classify_points(points)
    return data


def extract_point_from_np(data: dict, pos: int) -> list:
    return (data['x'][pos], data['y'][pos])


def classify_with_regions(regions, cssdata, time_independent_data, edge_data, css_regions):
    """
        Assigns cross-section id's based on region polygons.
        Within a region, assignment will be done by k nearest neighbour
    """
    

    time_independent_data['sclass'] = time_independent_data['region']
    #edge_data['sclass'] = edge_data['region']

    # Nearest Neighbour within regions
    for region in np.unique(css_regions):
        # Select cross-sections within this region
        css_xy = cssdata['xy'][css_regions==region]
        css_id = cssdata['id'][css_regions==region]

        # Select 2d points within region
        node_mask = time_independent_data['region'] == region
        x_2d_node = time_independent_data['x'][node_mask]
        y_2d_node = time_independent_data['y'][node_mask]

        edge_mask = edge_data['region'] == region
        x_2d_edge = edge_data['x'][edge_mask]
        y_2d_edge = edge_data['y'][edge_mask]

        # Do Nearest Neighour
        neigh = _get_class_tree(css_xy, css_id)
        css_2d_nodes = neigh.predict(np.array([x_2d_node, y_2d_node]).T)        
        css_2d_edges = neigh.predict(np.array([x_2d_edge, y_2d_edge]).T)

        # Update data in main structures
        time_independent_data['sclass'][node_mask] = css_2d_nodes # sclass = cross-section id
        
        edge_data['sclass'][edge_mask] = css_2d_edges
    
    return time_independent_data, edge_data


def classify_without_regions(cssdata, time_independent_data, edge_data):
    # Create a class identifier to map points to cross-sections
    neigh = _get_class_tree(cssdata['xy'], cssdata['id'])

    # Expand time-independent dataset with cross-section names
    time_independent_data['sclass'] = neigh.predict(np.array([time_independent_data['x'], time_independent_data['y']]).T)

    # Assign cross-section names to edge coordinates as well
    edge_data['sclass'] = neigh.predict(np.array([edge_data['x'], edge_data['y']]).T)

    return time_independent_data, edge_data


def get_fm2d_data_for_css(classname, dti, edge_data, dtd):
    """
    create a dictionary that holds all the 2D data for the cross-section with name 'classname'
    """
    x = dti['x'][dti['sclass'] == classname]
    y = dti['y'][dti['sclass'] == classname]
    area = dti['area'][dti['sclass'] == classname]
    region = dti['region'][dti['sclass'] == classname]
    islake = dti['islake'][dti['sclass'] == classname]
    waterdepth = dtd['waterdepth'][dti['sclass'] == classname]
    waterlevel = dtd['waterlevel'][dti['sclass'] == classname]
    vx = dtd['velocity_x'][dti['sclass'] == classname]
    vy = dtd['velocity_y'][dti['sclass'] == classname]
    face_section = dti['section'][dti['sclass'] == classname]
    # find all chezy values for this cross section, note that edge coordinates are used
    chezy = dtd['chezy_edge'][edge_data['sclass'] == classname]
    try:
        edge_faces = edge_data['edge_faces'][edge_data['sclass'] == classname]
    except KeyError:
        edge_faces = None
    #edge_nodes = edge_data['edge_nodes'][edge_data['sclass'] == classname]
    edge_x = edge_data['x'][edge_data['sclass'] == classname]
    edge_y = edge_data['y'][edge_data['sclass'] == classname]
    edge_section = edge_data['section'][edge_data['sclass'] == classname]  # roughness section number 

    # retrieve the full set for face_nodes and area, needed for the roughness calculation
    #face_nodes = edge_data['face_nodes'][dti['sclass'] == classname]
    #face_nodes_full = edge_data['face_nodes']
    area_full = dti['area']
    bedlevel_full = dti['bedlevel']
    bedlevel = dti['bedlevel'][dti['sclass'] == classname]

    velocity = (vx**2+vy**2)**0.5
    waterlevel[waterdepth == 0] = np.nan

    return_dict = {
        'x': x, 
        'y': y, 
        'area': area,
        'bedlevel': bedlevel, 
        'bedlevel_full': bedlevel_full, 
        'waterdepth': waterdepth, 
        'waterlevel': waterlevel, 
        'velocity': velocity, 
        'section': face_section,
        'chezy': chezy, 
        'region': region,
        'islake': islake,
        'edge_faces': edge_faces,
        'edge_x': edge_x,
        'edge_y': edge_y,
        'edge_section': edge_section,
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
    (_, index) = nn.kneighbors(location_array)

    # retrieve cell characteristic waterdepth
    centre_depth = waterdepth.iloc[index[0]]
    centre_level = waterlevel.iloc[index[0]]

    # remove nan values
    centre_depth[np.isnan(centre_depth)] = np.nanmin(centre_depth)
    centre_level[np.isnan(centre_level)] = np.nanmin(centre_level)
    
    return centre_depth.values[0], centre_level.values[0]

def get_extra_total_area(waterlevel, crest_level, transition_height, hysteresis=False):
    """
    releases extra area dependent on waterlevel using a logistic (sigmoid) function
    """
    return 1/(1+np.e**(np.log(0.00001)/(transition_height)*(waterlevel-(crest_level+0.5*transition_height))))

def return_volume_error(predicted, measured, gof='rmse'):
    non_nan_mask = ~np.isnan(predicted) & ~np.isnan(measured)
    predicted = predicted[non_nan_mask]
    measured = measured[non_nan_mask]
    error = np.array(predicted - measured)/np.maximum(np.array(measured), np.ones(len(measured)))
    return np.sum(error**2)

def interpolate_roughness(cross_section_list):
    """
    Creates a uniform matrix of z/chezy values for all cross-sections by linear interpolation
    """
    all_sections = [s for css in cross_section_list for s in css.friction_tables.keys()]

    sections = np.unique(all_sections)

    zstep = 0.1

    for section in sections:
        minimal_z = 1e20
        maximal_z = -1e20
        for css in cross_section_list:
            if section in list(css.friction_tables.keys()):
                minimal_z = np.min([minimal_z,
                                    np.min(css.friction_tables.get(section).level)])
                maximal_z = np.max([maximal_z,
                                    np.max(css.friction_tables.get(section).level)])

        for css in cross_section_list:
            if section in list(css.friction_tables.keys()):
                minmaxrange = np.arange(minimal_z, maximal_z, zstep)
                css.friction_tables.get(section).interpolate(minmaxrange)

def empirical_ppf(qs, p, val=None, single_value=False):
    """
    Constructs empirical cdf, then draws quantile by linear interpolation
    qs : array of quantiles (e.g. [2.5, 50, 97.5])
    p : array of random inputs

    return
    """
    if val is None: 
        p, val = get_empirical_cdf(p)

    if not single_value:
        output = list()
        for q in qs:
            output.append(np.interp(q / 100., p, val))
    else:
        output = np.interp(qs / 100., p, val)
    return output

def get_empirical_cdf(sample, n=100, method=1, ignore_nan=True):
    """
    Returns an experimental/empirical cdf from data. 

    Arguments:

        p : list

    Returns:

        (x, y) : lists of values (x) and cumulative probability (y)

    """

    sample = np.array(sample)
    if ignore_nan:
        sample = sample[~np.isnan(sample)]

    n = len(sample)
    val = np.sort(sample)
    p = np.array(range(n)) / float(n)

    return p, val

# endregion

# region // protected functions
def _read_fm_model(file_path):
    """input: FM2D map file"""
    fm_edge_keys = {'x': 'mesh2d_edge_x', 
                    'y': 'mesh2d_edge_y',
                    'edge_faces': 'mesh2d_edge_faces',
                    'edge_nodes': 'mesh2d_edge_nodes'
                    }
    edge_data = dict()
    # Open results file for reading
    res_fid = netCDF4.Dataset(file_path, 'r')

    # Time-invariant variables from FM 2D
    df = pd.DataFrame(columns=['x'], data=np.array(res_fid.variables['mesh2d_face_x']))
    df['y'] = np.array(res_fid.variables['mesh2d_face_y'])
    df['area'] = np.array(res_fid.variables['mesh2d_flowelem_ba'])
    df['bedlevel'] = np.array(res_fid.variables['mesh2d_flowelem_bl'])
    # These are filled later
    df['region'] = ['']*len(df['y'])  # 
    df['section'] = ['main']*len(df['y'])  # 
    df['sclass'] = ['']*len(df['y'])  # cross-section id
    df['islake'] = [False]*len(df['y'])  # roughness section number 

    

    # Edge data
    # edgetype 1 = 'internal'
    internal_edges = res_fid.variables['mesh2d_edge_type'][:] == 1
    for key, value in fm_edge_keys.items():
        try:
            edge_data[key] = np.array(res_fid.variables[value])[internal_edges]
        except KeyError:
            # 'edge_faces' does not always seem to exist in the file. 
            # todo: incorporate this function in its FmModelData with logger to
            # output a warning. For now, the omission of 'edge_faces' is handled
            # in FmModelData.
            pass

    edge_data['sclass'] = np.array(['']*np.sum(internal_edges), dtype='U99')
    edge_data['section'] = np.array(['main']*np.sum(internal_edges), dtype='U99')
    edge_data['region'] = np.array(['']*np.sum(internal_edges), dtype='U99')

    # node data (not used?)
    df_node = pd.DataFrame(columns=['x'], data=np.array(res_fid.variables['mesh2d_node_x']))
    df_node['y'] = np.array(res_fid.variables['mesh2d_node_y'])
    
    # Time-variant variables
    time_dependent = {
                      'waterdepth': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_waterdepth']).T, columns=res_fid.variables['time']),
                      'waterlevel': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_s1']).T, columns=res_fid.variables['time']),
                      'chezy_mean': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_czs']).T, columns=res_fid.variables['time']),
                      'chezy_edge': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_cftrt']).T[internal_edges], columns=res_fid.variables['time']),
                      'velocity_x': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_ucx']).T, columns=res_fid.variables['time']),
                      'velocity_y': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_ucy']).T, columns=res_fid.variables['time']),
                      'velocity_edge': pd.DataFrame(data = np.array(res_fid.variables['mesh2d_u1']).T, columns=res_fid.variables['time'])
                      }

    return df, edge_data, df_node, time_dependent



def _get_class_tree(xy, c):
    X = xy
    y = c
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X, y)
    return neigh

def _interpolate_roughness_css(cross_section, alluvial_range, nonalluvial_range):
    # change nan's to zeros
    chezy_alluvial = np.nan_to_num(cross_section.alluvial_friction_table[1])
    chezy_nonalluvial = np.nan_to_num(cross_section.nonalluvial_friction_table[1])

    waterlevel_alluvial = cross_section.alluvial_friction_table[0]
    waterlevel_nonalluvial = cross_section.nonalluvial_friction_table[0]

    # remove trailing zeros
    chezy_alluvial_trimmed = np.trim_zeros(chezy_alluvial)
    chezy_nonalluvial_trimmed = np.trim_zeros(chezy_nonalluvial)

    alluvial_nonzero_mask = chezy_alluvial.to_numpy().nonzero()[0]
    nonalluvial_nonzero_mask = chezy_nonalluvial.to_numpy().nonzero()[0]

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
