# Overview

FM2PROF is a tool to generate 1D cross-sectional information from 2D data. It supports two modes:

1. Elevation only mode by providing a DEM
2. Hydraulic mode by providing a 2D hydrodynamic simulation. 

In the second mode, FM2PROF uses flow patterns from the 2D simulation to distinguish between active flow zones and dead zones, compute effective friction values, and generate storage-conveyance relationships that accurately downscale 2D dynamics to a 1D model. 

FM2PROF uses a nearest-neighbour approach to assign 2D information to 1D locations. Finer grain control over this assignment is possible (but optional) by providing polygons. 

To get started, see the [installation guide](installation.md) and [quick start tutorial](quickstart.md), or explore the [configuration options](configuration.md) to customise the extraction process for your specific modelling needs.