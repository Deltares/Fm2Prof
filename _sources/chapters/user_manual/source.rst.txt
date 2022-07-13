.. _pythonAPI:

Python API
======================================

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


API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fm2prof.Project
   :members:
   :show-inheritance:
   :inherited-members: 

.. _SourceUtils:

Utilities
------------

.. automodule:: fm2prof.utils
   :members: 


