.. _ChapterInstallation:


Getting |project|
=========================

There are several ways to build or install |project|, depending on your use case. 

Executable
...............

|project| is compiled to an executable for all major operating systems (version 1.5.4 and higher). Download them from `Releases <https://github.com/Deltares/Fm2Prof>`_ on the Github home page. 

The executable provides a command line interface (CLI). To use it, make sure it is available in your Path. To view available options, open a Terminal and type:

.. code-block:: shell

	FM2PROF --help

.. warning::
	The first time you call the executable, it may take a long time to produce results. This is because your OS needs to unpack the executable. 

To start a new project in the current working directory, type:

.. code-block:: shell

	FM2PROF create MyProject

This will create a configuration file with all default options. At minimum, you will have to provide a path to a 2D simulation and to a :ref:`CrossSectionLocationFile`

To perform a validity check on your project, run:

.. code-block:: shell

	FM2PROF check MyProject

If you have not provided input files in the configuration file, this will report errors pointing to this. If there are no errors, you can proceed to run your project:

.. code-block:: shell

	FM2PROF run MyProject

The executable uses the CLI API. For further reference, see :ref:`cli documentation`



Python package
...............

Alternatively, you can run |project| as a Python package. The advantage of this is that you have access to utilities and experimental methods that have not (yet) been implemented in the CLI api. The disadvantage is that you will have to setup a Python environment. 


Installation from local checkout
--------------------------------

Using poetry
^^^^^^^^^^^^

Activate a python 3.7 virtual environment. Next, checkout |project| from the git repository and install with poetry

.. code-block:: shell

	git clone https://github.com/deltares/fm2prof
	cd fm2prof
	poetry install

Now, you can import FM2PROF as any regular python package. For example, to launch a VisualStudioCode instance:

.. code-block:: shell

	poetry shell
	code .

You can also run fm2prof using the :ref:`cli documentation`:

.. code-block:: shell

	poetry run f2p --help


Using conda
^^^^^^^^^^^^

It is highly recommended to create a new, separate environment for FM2PROF. To create a new environment, open an (Anaconda) prompt and navigate to the directory of the python source code, and type: 

.. code-block:: shell

	git clone https://github.com/deltares/fm2prof
	cd fm2prof
	conda env create -f environment.yml


This will create a new Python 3.7 environment called "fm2prof", with necessary packages installed. To install FM2PROF:

.. code-block:: bash

	conda activate fm2prof
	conda develop .

You can now use FM2PROF as a Python package using :ref:`source documentation` or the :ref:`cli documentation`.

Installation from PyPI 
---------------------------------

PyPi is currently not supported

