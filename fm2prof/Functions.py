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
import csv
import os

# region // imports
import netCDF4
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

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


def classify_roughness_sections_by_polygon(sections, data, logger):
    """assigns edges to a roughness section based on polygon data"""
    logger.debug("....gathering points")
    points = [(data["x"][i], data["y"][i]) for i in range(len(data["x"]))]
    logger.debug("....classifying points")
    data["section"] = sections.classify_points(points)
    return data


def extract_point_from_np(data: dict, pos: int) -> list:
    return (data["x"][pos], data["y"][pos])


def classify_with_regions(
    regions, cssdata, time_independent_data, edge_data, css_regions
):
    """
    Assigns cross-section id's based on region polygons.
    Within a region, assignment will be done by k nearest neighbour
    """

    time_independent_data["sclass"] = time_independent_data["region"].astype(str)
    # edge_data['sclass'] = edge_data['region']

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
        neigh = _get_class_tree(css_xy, css_id)
        css_2d_nodes = neigh.predict(np.array([x_2d_node, y_2d_node]).T)
        css_2d_edges = neigh.predict(np.array([x_2d_edge, y_2d_edge]).T)

        # Update data in main structures
        time_independent_data.loc[node_mask, "sclass"] = css_2d_nodes  # sclass = cross-section id

        edge_data["sclass"][edge_mask] = css_2d_edges

    return time_independent_data, edge_data


def classify_without_regions(cssdata, time_independent_data, edge_data):
    # Create a class identifier to map points to cross-sections
    neigh = _get_class_tree(cssdata["xy"], cssdata["id"])

    # Expand time-independent dataset with cross-section names
    time_independent_data["sclass"] = neigh.predict(
        np.array([time_independent_data["x"], time_independent_data["y"]]).T
    )

    # Assign cross-section names to edge coordinates as well
    edge_data["sclass"] = neigh.predict(np.array([edge_data["x"], edge_data["y"]]).T)

    return time_independent_data, edge_data


def mirror(array, reverse_sign=False):
    """
    Mirrors array

    :param array:
    :param reverse_sign:
    :return:
    """
    if reverse_sign:
        return np.append(np.flipud(array) * -1, array)
    else:
        return np.append(np.flipud(array), array)


def get_centre_values(location, x, y, waterdepth, waterlevel):
    """
    Find output point closest to x,y location, output depth and water level as nd arrays

    """
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(np.array([x, y]).T)

    # conversion to 2d array, as 1d arrays are deprecated for kneighbors
    location_array = np.array(location).reshape(1, -1)
    (_, index) = nn.kneighbors(location_array)

    # retrieve cell characteristic waterdepth
    centre_depth = waterdepth.iloc[index[0]].to_numpy()
    centre_level = waterlevel.iloc[index[0]].to_numpy()

    # When starting from a dry bed, the centre_level may have nan values
    #
    bed_level = np.nanmin(centre_level - centre_depth)
    # centre_depth[np.isnan(centre_depth)] = np.nanmin(centre_depth)
    centre_level[np.isnan(centre_level)] = bed_level

    return centre_depth[0], centre_level[0]


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
            output.append(np.interp(q / 100.0, p, val))
    else:
        output = np.interp(qs / 100.0, p, val)
    return output


def get_empirical_cdf(sample, method=1, ignore_nan=True):
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
        waterlevel_alluvial_trimmed = waterlevel_alluvial[
            alluvial_nonzero_mask[0] : alluvial_nonzero_mask[-1] + 1
        ]
        alluvial_interp = np.interp(
            alluvial_range, waterlevel_alluvial_trimmed, chezy_alluvial_trimmed
        )

        # assign
        cross_section.alluvial_friction_table[0] = alluvial_range
        cross_section.alluvial_friction_table[1] = pd.Series(data=alluvial_interp)

    if np.sum(nonalluvial_nonzero_mask) > 0:
        waterlevel_nonalluvial_trimmed = waterlevel_nonalluvial[
            nonalluvial_nonzero_mask[0] : nonalluvial_nonzero_mask[-1] + 1
        ]
        nonalluvial_interp = np.interp(
            nonalluvial_range, waterlevel_nonalluvial_trimmed, chezy_nonalluvial_trimmed
        )

        # assign
        cross_section.nonalluvial_friction_table[0] = nonalluvial_range
        cross_section.nonalluvial_friction_table[1] = pd.Series(data=nonalluvial_interp)


# endregion
