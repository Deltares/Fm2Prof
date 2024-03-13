"""
FM2PROF

Importing region or sections to grid for use in FM2PROF. 
Preprocessing step to be executed in DeltaShell. 

Works with: 2020.02
Known issues with: 2021.04

Python 2.7 syntax!

# USAGE:
	
	1. Start Delft3D Flexible Mesh Suite GUI
	2. New model -> Flow Flexible Mesh Model
	3. In Project Pane, Right-click on Grid, import, choose FM2PROF results file. 
	4. View -> show toolbox
	5. In Toolbox pane, create new script and copy the contents this file
	6. Change the variable _gridFile, _polyFile and _logFile to the appropriate values
	7. Run Script (This will take a while)

"""

import json
import os
import sys
import time
from datetime import datetime

import DeltaShell.Plugins.SharpMapGis.ImportExport as ie
import Libraries.FlowFlexibleMeshFunctions as fmf
import Libraries.SpatialOperations as so
import Libraries.StandardFunctions as sf

# Set PATHS HERE
_gridFile = r"c:\Users\berend_kn\projects\2017_fm2prof\_dev\src\trunk\Maas_dir\Model_FM\Maas_2_merged_map.nc"
_polyFile = r"c:\Users\berend_kn\projects\2017_fm2prof\_dev\src\trunk\Maas_dir\input\section_polygon.geojson"
_logFile = r"c:\Users\berend_kn\projects\2020_03_fm2prof\03_model_development\m3\f2p_ds_log.txt"

# region helper functions
def ReadPolygonFromFile(filename):
    with open(filename, "r") as f:
        poly = json.load(f)

    features = poly.get("features")
    print("Polygon consists of {} features".format(len(features)))
    for feature in features:
        yield {
            "name": feature.get("properties").get("name"),
            "id": feature.get("properties").get("id"),
            "coords": feature.get("geometry").get("coordinates")[0],
        }


def CreateModel():
    # Create a Delft3D flow model
    model = fmf.WaterFlowFMModel()

    # Add model to current project
    sf.AddToProject(model)

    return model


def ImportGridToModel(model):
    model.Grid = ie.NetFileImporter.ImportGrid(_gridFile)
    return model


def AssignToBathymetry():
    operationName = "SetValue"
    model = sf.GetModelByName("FlowFM")
    object = model.Bathymetry
    polygon = [[185714, 349311], [187450, 349352], [157510, 348026], [185653, 348077]]

    with open(_logFile, "w") as f:
        start = time.time()
        f.write("FM2PROF Preprocessing tool\n")
        f.write("Writing region polygons to bathymetry\n")
        f.write("Setting default value to -999\n")
        so.AddSpatialOperationByPolygon(
            operationName,
            model,
            object,
            "Envelope",
            values=[-999],
            pointwiseType="Overwrite",
        )
        f.write("done in {:.1f} seconds\n".format(time.time() - start))

    # Loop through polygons
    for i, polygon in enumerate(ReadPolygonFromFile(_polyFile)):
        # Each polygon should have a field named 'id', which is an integer. This integer is assigned to the bathymetry
        poly_id = int(polygon.get("id"))

        with open(_logFile, "a") as f:
            start = time.time()
            f.write(
                ("Assigning name: {}, id {} to bathymetry with value {}\n").format(
                    polygon.get("name"), poly_id, poly_id
                )
            )
            so.AddSpatialOperationByPolygon(
                operationName,
                model,
                object,
                polygon.get("coords")[0],
                values=[poly_id],
                pointwiseType="Overwrite",
            )
            f.write("done in {:.1f} seconds\n".format(time.time() - start))


# endregion
AssignToBathymetry()
