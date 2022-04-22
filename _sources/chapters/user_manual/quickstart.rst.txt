Quickstart 
====================


.. note::
    We assume you have a conda environment ``fm2prof`` and fm2prof installed following the :ref:`ChapterInstallation`

In this tutorial you'll walk through the basics of setting up a project with |project|. 


Input files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With |project| you can generate input for a 1D model from a 2D model. The basic workflow of |project| is shown below. The specific steps are detailed below. 

.. figure:: ../figures/basic_workflow.png
    :align: center
    :alt: alternate text
    :figclass: align-center

    A minimal |project| workflow



2D Model
~~~~~~~~~~~~~~~


This is the 2D model to be emulated. |project| is agnostic about the 2D model software, but requires UGRID style output. 


River models
............

In rivers, compartimentalisation may cause some part of the floodplains to flow at a later stage, or not flow at all. If cross-sections are only based on geometry, this may lead to overestimation of the cross-sectional area. |project| can use information from a 2D model computation to improve the quality of the cross-sections, by computing which cells contribute to flow at given water levels. This is best done with a special simulation using monotically rising conditions:

* a slowly rising discharge at the upstream boundary
* a slowly rising waterlevel at the downstream boundary

Ideally, the water slope remains constant along the channel for the entire computation, i.e. no significant backwater effects. 

Lake models
............

*Lake models are not yet tested/supported* 


1D Network
~~~~~~~~~~~~~~~~

A 1D network should already be available. At the least, you should know the locations at which cross-sectional and roughness information should be derived.

.. note::
    If your 1D model is a SOBEK 3 model, the CrossSectionLocationFile can be derived from the Network Definition file (DIMR configuration) using :meth:`fm2prof.utils.networkdeffile_to_input`.


Running |project|
^^^^^^^^^^^^^^^^^
There are currently two ways to run FM2PROF: by using a terminal or with the Python API. 


Using |project| with a terminal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. note::
    The terminal interface is also referred to as the CLI (command line interface)

First, open a command window or powershell and navigate to `<fm2prof source code>/bin`, which contains `fm2profConsole.py`. 

To start a new project:

.. code-block:: bash
    
    # while in ./fm2prof/bin
    python ./fm2profConsole.py -n EmptyProject

this will create a new file called 'EmptyProject.ini'. This is a valid configuration file with all parameters set to their default values. In this file, manually add at least the following settings: 

.. code-block:: text
    
    [input]
    2DMapOutput                   =           # Output file from FM2D model (.net file)
    CrossSectionLocationFile      =           # .csv or .txt file (, as delimiter) that contains four columns: X_coordinate,Y_coordinate,BranchName,Length,Chainage.

For this tutorial, use the following test data:

.. code-block:: text
    
    [input]
    2DMapOutput                   = ../tests/external_test_data/case_01_rectangle/Data/2DModelOutput/FlowFM_map.nc
    CrossSectionLocationFile      = ../tests/external_test_data/case_01_rectangle/Data/cross_section_locations.xyz

Next, load the project:

.. code-block:: bash

    python fm2profConsole.py -f EmptyProject.ini

Check the results whether everything is ok. You should also see a print-out of the configuration file. These are the settings that will be used and can be altered using a config file. The configuration file does not need all parameters to be specified. If a parameter is not in the configuration file, default values will be used. 

To generate output, use: 

.. code-block:: bash

    python fm2profConsole.py -f EmptyProject.ini -r

Using the Python API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FM2PROF provides modules that can be imported in any Python script or interactive console (e.g. a Jupyter notebook). 

.. code-block:: python
    
    # Use the project class
    from fm2prof import Project

    # To initialise a project without configuration
    project = Project()

If you initialise a project without an argument, a configuration is loaded from template. You an use this project object to change the input files and parameters:

.. code-block:: python
    
    # Note that all paths are relative to os.getcwd!
    project.set_inputfile('2DMapOutput', 'tests/external_test_data/case_01_rectangle/Data/2DModelOutput/FlowFM_map.nc')
    project.set_inputfile('CrossSectionLocationFile', 'tests/external_test_data/case_01_rectangle/Data/cross_section_locations.xyz')

To run the project:

.. code-block:: python
    
    project.run()

See :ref:`source documentation` for a full overview of available methods. 