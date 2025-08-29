"""Contains functions used for the emulation/reduction of 2D models to 1D models for Delft3D FM (D-Hydro)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

if TYPE_CHECKING:

    import pandas as pd


def classify_with_regions(
    cssdata: dict,
    time_independent_data: pd.DataFrame,
    edge_data: dict,
    css_regions: list[str],
    ) -> tuple[pd.DataFrame, dict]:
    """Assign cross-section id's based on region polygons.

    Within a region, assignment will be done by k nearest neighbour
    """
    time_independent_data["sclass"] = time_independent_data["region"].astype(str)

    # Nearest Neighbour within regions
    for region in time_independent_data["region"].unique():
        # Select cross-sections within this region
        css_xy = cssdata["xy"][np.array(css_regions) == region]
        css_id = cssdata["id"][np.array(css_regions) == region]

        if len(css_id) == 0:
            # no cross-sections in this region
            continue
        # Select 2d points within region
        node_mask = time_independent_data["region"] == region
        x_2d_node = time_independent_data["x"][node_mask]
        y_2d_node = time_independent_data["y"][node_mask]

        edge_mask = np.array(edge_data["region"]) == region
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

def classify_without_regions(
    cssdata: dict,
    time_independent_data: pd.DataFrame,
    edge_data: dict,
    ) -> tuple[pd.DataFrame, dict]:
    """Classify without regions."""
    # Create a class identifier to map points to cross-sections
    neigh = _get_class_tree(cssdata["xy"], cssdata["id"])

    # Expand time-independent dataset with cross-section names
    time_independent_data["sclass"] = neigh.predict(
        np.array([time_independent_data["x"], time_independent_data["y"]]).T,
    )

    # Assign cross-section names to edge coordinates as well
    edge_data["sclass"] = neigh.predict(np.array([edge_data["x"], edge_data["y"]]).T)

    return time_independent_data, edge_data

def get_centre_values(
    location: np.array,
    x: float,
    y: float,
    waterdepth: pd.DataFrame,
    waterlevel: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Find output point closest to x,y location, output depth and water level as nd arrays."""
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
    centre_level[np.isnan(centre_level)] = bed_level

    return centre_depth[0], centre_level[0]

def _get_class_tree(xy: np.ndarray, c: np.ndarray) -> KNeighborsClassifier:
    x = xy # {array-like, sparse matrix} of shape (n_samples, n_features)
    y = c # {array-like, sparse matrix} of shape (n_samples,)
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(x, y) 
    return neigh

