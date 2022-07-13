Files
============

Input files 
+++++++++++++

Configuration file
-------------------

.. program-output:: python -c "from fm2prof.IniFile import IniFile; print(IniFile().print_configuration())"


.. _2DMapOutput:	

2DMapOutput
-------------------------
`netCDF` file with UGRID convention

A simulation performed with the 2D model to be reduced to 1D. |project| is agnostic about the 2D model software, but requires UGRID style output from a hydraulic simulation.

The simulation must be such that it ensures a a monotonically rising water level. Ideally, the water slope remains constant along the channel for the entire computation, i.e. no significant backwater effects. 

	In rivers, compartimentalisation may cause some part of the floodplains to flow at a later stage, or not flow at all. If cross-sections are only based on geometry, this may lead to overestimation of the cross-sectional area. |project| can use information from a 2D model computation to improve the quality of the cross-sections, by computing which cells contribute to flow at given water levels. This is best done with a special simulation using monotically rising conditions:

	* a slowly rising discharge at the upstream boundary
	* a slowly rising waterlevel at the downstream boundary


.. _CrossSectionLocationFile:	

CrossSectionLocationFile
-------------------------
Comma-seperated file. 
Syntax: `x, y, branch name, cross-section length, chainage`

This file specifies at which locations a 1D cross-section and roughness information must be generated. 

The CrossSectionLocationFile can be generated from a SOBEK 3 Network Definition file (DIMR configuration) using :meth:`fm2prof.utils.networkdeffile_to_input`.

.. note::
    The methods in utils are currently not supported in the CLI API. You will need to install FM2PROF as a python package. 


Where:
	
	x, y [float]: coordinates of the cross-section. 

	branch name [str]: name of the 1D branch on which the cross-section is defined

	cross-section length [float]: see :term:`Control volumes`

	chainage [float]: location on branch
	
Example:

.. code-block:: text

	0,75,case2,250,0
	500,75,case2,500,500
	1000,75,case2,500,1000
	1500,75,case2,500,1500
	2000,75,case2,500,2000
	2500,75,case2,500,2500
	3000,75,case2,250,3000


.. _PolygonFiles

PolygonFiles
-------------------------
`geojson` multi-polygon files. 

Files to distinguish regions and sections. See :term:`Region` and :term:`Section` for more information. 

.. warning::
	Donot-polygons are not supported
	

.. warning::
	In the current version, |project| reads and validates the polygon files, but does not use them to allocate regions and sections. This is a very computationally extensive task. Therefore, you will need to pre-process this step. Currently, this preprocessing step is only supported through a DeltaShell script that is distributed with this repository. 

.. _outputFiles:


.. _branchRuleFile:	

Branch rule file
-------------------------

Optional input for :class:`fm2prof.utils.GenerateCrossSectionLocationFile`


Output files
+++++++++++++

SOBEK 3 (DIMR)
-------------------------
The following files can be copied directly to the `dflow1d` directory of the SOBEK 3 model. 

	- CrossSectionDefinitions.ini 
	- CrossSectionLocations.ini 
	- roughness-Main.ini
	- roughness-FloodPlain1.ini

SOBEK 3 (<3.5, DeltaShell)
---------------------------
Older versions of SOBEK 3 may import the following files through the user interface (DeltaShell):

	- geometry.csv
	- roughness.csv

Diagnostic files
-----------------
|project| produces a number of files that can be used to diagnose its output.

**fm2prof.log**
	The log file containing error, warning and debug messages. This file is always generated. 

**volumes.csv**
	Comma-seperated file that contains intermediate results. It is used by `fm2prof.utils` to produce figures (e.g. see :ref:`diagnosevisualiseOutput`)

**edge_output.geojson & face_output.geojson**
	point-data in geojson format. Useful to visualise which 2D grid cells are assigned to which cross-section. Only produced if `ExportMapFiles=True`

