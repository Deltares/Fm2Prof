.. _diagnosis:

Troubleshooting
====================

After running |project|, it is important to check its output. In this section we provide some steps to diagnose and interprete your model results. In short, the steps are the following:

1. Check the log for errors & warnings 
2. Visualising output
3. Comparing model results

.. tip::

    |project| output consists of:

        - SOBEK 3 format files (`geometry.csv`, `roughness.csv`)
        - DFlow1D format files (`CrossSectionLocations.ini`, `CrossSectionDefinitions.ini`, etc. )
        - diagnosis files (`fm2prof.log`, `volumes.csv`, `cross_section_volumes.geojson`)
        - optional diagnosis files (see the `ExportMapFiles` parameter)
        - the log file (fm2prof.log)


1. Checking the log 
....................

|project| logs to the console and to a file. By default, they do not show the same information: messages flagged as `debug` are not written to screen by default (but they are to file). The log is written to the output directory as `fm2prof.log` by default. The log first prints the configuration used during the run. You can use this to reproduce your results later one. 

.. hint::

    It is good practice to open the log and `Ctrl-F` search for warnings and errors. Warnings alert the user on possible input error (e.g. overlapping polygons), but that do not prevent execution of the problem, and may even not affect the output adversely. Errors should not be ignored, but resolved. 

In general, the log consists of the following sections:

.. code-block:: text

    2020-07-23 15:55    INFO    0% T+ 0.04s Opening FM Map file
    2020-07-23 15:55    INFO    0% T+ 0.28s Closed FM Map file
    2020-07-23 15:55    INFO    0% T+ 0.28s Opening css file
    2020-07-23 15:55    INFO    0% T+ 0.28s Closed css file
    2020-07-23 15:55    INFO    0% T+ 0.28s All 2D points assigned to the same region and classifying points to cross-sections
    2020-07-23 15:55    INFO    0% T+ 0.34s Assigning point to sections without polygons
    2020-07-23 15:55    INFO    0% T+ 0.35s finished reading FM and cross-sectional data data
    2020-07-23 15:55   DEBUG    0% T+ 0.35s Number of: timesteps (169), faces (360), edges (654)

If |project| fails in this section, the initialisation or files may not be loaded properly. This phase may take some minutes for large models, especially if regions and sections are used. 

.. tip::

    The number of faces and edges corresponds to the 2D grid cells. They greatly determines the initialisation time. For example, the initialisation step for the Meuse model (face count: 353,882) take about 60-70 seconds. 

    The number of timesteps determines the time for each cross-section to be derived (next section). 



Next, the program will loop through all the cross-sections. A single iteration may look like this (debug information is cropped out):

.. code-block:: text

    2020-07-23 15:55    INFO    0% T+ 0.36s Starting new task: case1_0
    2020-07-23 15:55    INFO    0% T+ 0.66s Initiated new cross-section
    2020-07-23 15:55    INFO    0% T+ 0.87s Retrieving centre point values
    2020-07-23 15:55    INFO    0% T+ 0.88s Identifying lakes
    2020-07-23 15:55    INFO    0% T+ 1.49s Seperating flow from storage
    2020-07-23 15:55    INFO    0% T+ 2.04s Computing cross-section from water levels
    2020-07-23 15:55    INFO    0% T+ 2.04s Computing cross-section from bed levels
    2020-07-23 15:55    INFO    0% T+ 2.07s correction finished
    2020-07-23 15:55    INFO    0% T+ 2.07s Cross-section reduced from 173 to 20 points
    2020-07-23 15:55    INFO    0% T+ 2.08s Computed roughness
    2020-07-23 15:55    INFO    0% T+ 2.08s Cross-section case1_0 derived in 1.73 s


.. tip::
    
    If each cross-section takes a long time to derive, the number of output timesteps in your 2D model may be too large. Each cross-section in the Meuse model take < 2 second to be derived. 


Finally the program will finish:

.. code-block:: text 

    2020-06-05 08:04    INFO    0% T+ 692.61s Starting new task: Finalizing
    2020-06-05 08:04    INFO    0% T+ 692.61s Interpolating roughness
    2020-06-05 08:04    INFO    0% T+ 692.68s Export model input files to ../Maas_dir/Output/maas_default_settings\test19
    2020-06-05 08:04    INFO    0% T+ 698.96s Exported output files, FM2PROF finished
    2020-06-05 08:04    INFO    0% T+ 698.97s Export geojson output to ../Maas_dir/Output/maas_default_settings\test19
    2020-06-05 08:04   DEBUG    0% T+ 698.99s Collected points, dumping to file
    2020-06-05 08:04   DEBUG    0% T+ 729.79s Done
    2020-06-05 08:04   DEBUG    0% T+ 729.86s Collected points, dumping to file
    2020-06-05 08:05   DEBUG    0% T+ 773.85s Done

.. hint::
    
    The Meuse pilot model take about 13 minutes to produce 1D input. 



2. Visualising output 
......................

There are various ways to visualise |project| output. 

With utils.VisualiseOutput
---------------------------


First, the `fm2prof.utils` module provides a class to directly load and visualise the output:

.. code-block:: python

    from fm2prof import utils

    output_dir = 'tests\Output\RunWithFiles_Output\case_05_dyke\CaseName01'

    viz = utils.VisualiseOutput(output_dir)

    for css in viz.cross_sections():
        # This creates a directory 'figures' in output_dir
        viz.make_figure(css)

This produces figures in the output directory. 

.. figure:: ../figures/interprete_figure.png
    :align: center
    :alt: alternate text
    :figclass: align-center
    :width: 100%

    Example interpretation of a ` fm2prof.utils.VisualiseOutput`  figure for testcase :ref:`validation_summerdike`. 

The shape of the volume graph depends on the shape of the geometry:


.. figure:: ../figures/volume_graphs_types.png
    :align: center
    :alt: alternate text
    :figclass: align-center
    :width: 50%

    The shape of the volume graph for given geometries

For most real-world rivers, the volume graphs will follow a compound 'trapezoidal' model, with one or several 'compartimentalisation' effects. 


With GIS software
---------------------------

A second way to visualise is with GIS software (such as QGIS). By default two files can be readily loaded in GIS software: the cross-section input file (which is a text delimited format) and the `cross_section_volumes.geojson` file. The latter is a convex-hull approximation of the control volumes (see :ref:`terminology`).


.. figure:: ../figures/gis_visualisation_maas_01.png
    :align: center
    :alt: alternate text
    :figclass: align-center
    :width: 100%

    Visualisation of the cross-section location (input) file and the convex-hull approximation of the control volumes (cross_section_volumes.geojson). The overlap of the polygons follows from the convex hull approximation - overlap is not actually possible.  


Closer inspection is possible by setting `ExportMapFiles` to `True` in the configuration file. Note that this an create really large files. 

.. tip::

    For inspecting one or just a few cross-sections, use the `CssSelection` parameter to limit the output. 


.. figure:: ../figures/gis_visualisation_maas_02.png
    :align: center
    :alt: alternate text
    :figclass: align-center
    :width: 100%

    Visualisation of the actual assignment of 2D points to cross-sections using nearest-neighbour approximation. (note: region polygons were used here)


.. figure:: ../figures/gis_visualisation_maas_03.png
    :align: center
    :alt: alternate text
    :figclass: align-center
    :width: 100%

    Visualisation of the (automatically classified) :term:`lakes`. Be aware of the specific definition of 'lake'. Water bodies that are hydraulically connected to the main channel at low water depths are not flagged as 'lake'. 

.. figure:: ../figures/gis_visualisation_maas_04.png
    :align: center
    :alt: alternate text
    :figclass: align-center
    :width: 100%

    Visualisation of the roughness section. The red dots are 2D points that were not within any polygon in the `SectionPolygonFile`. These are automatically added to the main section. This will yield a warning in the log file. In general it is a good idea to expand the polygons to cover all files. 

