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
def read_fm2prof_input(res_file, css_file, regions, sections):
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
    (time_independent_data, edge_data, node_coordinates, time_dependent_data) = _read_fm_model(res_file)

    # Load locations and names of cross-sections
    cssdata = _read_css_xyz(css_file)

    # Regions
    if regions is not None:
        time_independent_data, edge_data = classify_with_regions(regions, cssdata, time_independent_data, edge_data)
    else:
        time_independent_data, edge_data = classify_without_regions(cssdata, time_independent_data, edge_data)

    if sections is not None:
        edge_data = classify_roughness_sections(sections, edge_data)

    return time_dependent_data, time_independent_data, edge_data, node_coordinates, cssdata

def classify_roughness_sections(sections, edge_data):
    """ assigns edges to a roughness section based on polygon data """
    points = [(edge_data['x'][i], edge_data['y'][i])
              for i in range(len(edge_data['x']))]
    edge_data['section'] = sections.classify_points(points)
    return edge_data

def classify_with_regions(regions, cssdata, time_independent_data, edge_data):
    """
        Assigns cross-section id's based on region polygons. 
        Within a region, assignment will be done by k nearest neighbour
    """
    css_regions = regions.classify_points(cssdata['xy'])

    xy_tuples_2d = [(time_independent_data.get('x').values[i], 
                     time_independent_data.get('y').values[i]) for i in range(len(time_independent_data.get('x')))]
    
    time_independent_data['region'] = regions.classify_points(xy_tuples_2d)

    xy_tuples_2d = [(edge_data.get('x')[i], 
                     edge_data.get('y')[i]) for i in range(len(edge_data.get('x')))]
    
    edge_data['region'] = regions.classify_points(xy_tuples_2d)

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

    # find all chezy values for this cross section, note that edge coordinates are used
    chezy = dtd['chezy_edge'][edge_data['sclass'] == classname]
    edge_nodes = edge_data['edge_nodes'][edge_data['sclass'] == classname]
    edge_x = edge_data['x'][edge_data['sclass'] == classname]
    edge_y = edge_data['y'][edge_data['sclass'] == classname]
    edge_section = edge_data['section'][edge_data['sclass'] == classname]  # roughness section number 

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
        'region': region,
        'islake': islake,
        'edge_nodes': edge_nodes, 
        'edge_x': edge_x,
        'edge_y': edge_y,
        'edge_section': edge_section,
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
    # These are filled later
    df['region'] = ['']*len(df['y'])  # 
    df['sclass'] = ['']*len(df['y'])  # cross-section id
    df['islake'] = [False]*len(df['y'])  # roughness section number 

    

    # Edge data
    # edgetype 1 = 'internal'
    internal_edges = res_fid.variables['mesh2d_edge_type'][:] == 1
    edge_data = {'x': np.array(res_fid.variables['mesh2d_edge_x'])[internal_edges],
                 'y': np.array(res_fid.variables['mesh2d_edge_y'])[internal_edges],
                'edge_nodes': np.array(res_fid.variables['mesh2d_edge_nodes'])[internal_edges],
                'face_nodes': np.array(res_fid.variables['mesh2d_face_nodes']),
                'sclass': np.array(['']*np.sum(internal_edges), dtype='U99'),
                'section': np.ones(np.sum(internal_edges)),
                'region': np.array(['']*np.sum(internal_edges), dtype='U99')
                }

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

def _read_css_xyz(file_path : str, delimiter = ','):
    
    if not file_path or not os.path.exists(file_path):
        raise IOError('No file path for Cross Section location file was given, or could not be found at {}'.format(file_path))
    
    with open(file_path, 'r') as fid:
        input_data = dict(xy=list(), id=list(), branchid=list(), length=list(), chainage=list())
        for line in fid:
            try:
                (cssid, x, y, length, branchid, chainage) = line.split(delimiter)
            except ValueError:
                # revert to legacy format
                (x, y, branchid, length, chainage) = line.split(delimiter)
                cssid = branchid + '_' + str(round(float(chainage)))

            input_data['xy'].append((float(x), float(y)))
            input_data['id'].append(cssid)
            input_data['length'].append(float(length))
            input_data['branchid'].append(branchid)
            input_data['chainage'].append(float(chainage))

        # Convert everything to nparray
        for key in input_data:
            input_data[key] = np.array(input_data[key])
        return input_data

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
