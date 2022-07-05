.. _ChapterQuickstart:

Tutorial 
========

.. note::
    In this tutorial we will use the :ref:`cli documentation`. You can either use the executable or the Python package. See :ref:`ChapterInstallation` to obtain either of these. 

In this tutorial you'll walk through the basics of creating a 1D model from a 2D model with |project|. 

To use |project|, we need at least two input files:

- :ref:`2DMapOutput`
- :ref:`CrossSectionLocationFile`

in this tutorial we will use the test data bundled with the package. You will find this in the source directory:

.. code-block:: text

    # 2DMapOutput
    tests\test_data\cases\case_02_compound\Data\2DModelOutput\FlowFM_map.nc 
    # CrossSectionLocationFile
    tests\test_data\cases\case_02_compound\Data\cross_section_locations.xyz  

This case is a simple 2D compound channel model. 

Create a new |project| project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To start a new project:

.. code-block:: shell
    
    FM2PROF create MyProject

This will create a new directory "MyProject". In this directory you will find `MyProject.ini`. This is a valid configuration file with all parameters set to their default values. You will also find an `input` and `output` directory.

Modify the input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For the purpose of this tutorial, move the 2D model simulation file to `input`. This is not a requirement - you can refer to any location on your pc. 

Open the configuration file with your favorite text editor and change the following two settings to your input data:

.. code-block:: text
    
    [input]
    2DMapOutput                   =         # Output file from FM2D model (.net file)
    CrossSectionLocationFile      =           # .csv or .txt file (, as delimiter) that contains four columns: X_coordinate,Y_coordinate,BranchName,Length,Chainage.


To be sure you used the correct data, run:

.. code-block:: shell

    FM2PROF check MyProject

Study the output. Errors indicate that something is wrong with your input. In that case, you will need to correct the configurationfile. 

Run FM2PROF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You should also see a print-out of the configuration file. These are the settings that will be used and can be altered using a config file. The configuration file does not need all parameters to be specified. If a parameter is not in the configuration file, default values will be used. 

To generate output, use: 

.. code-block:: shell

    FM2PROF run MyProject

All :ref:`outputFiles` are written to the `output` directory. 

Inspect the results
^^^^^^^^^^^^^^^^^^^^^

After generating output it is important to check whether everything went well. |project| provides various tools and output files to do this. Go to :ref:`diagnosis` and try to generate figures with `fm2prof.utils`. 