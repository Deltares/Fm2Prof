.. _ChapterInstallation:


Installation Instructions
=========================

Requirements
---------------------------------

Minimum requirements
^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    Some requirements are available from multiple sources and distributions. We recommend using `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage the python environment 

* `Python <http://www.python.org/>`_ ** == 3.6**. The conda distribution is highly recommended. Python 2 syntax is not supported. The software is developed on 3.6, but earlier versions may work as well. 


Optional Requirements
^^^^^^^^^^^^^^^^^^^^^^^^

* **QGIS** to visualise FM2PROF output


Installation instructions
---------------------------------

.. note::
	This guide assumes conda is installed



Obtaining the python source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get the latest version of the code, you need to have a Subversion (SVN) client installed. If you have SVN installed with cli option, open a command prompt/terminal and type:

.. code-block:: bash

	svn checkout https://repos.deltares.nl/repos/RIVmodels/rivtools/branches/fm2profTool

Altenatively, use a GUI (like TortoiseSVN) to do this. 


.. _SectionPythonEnv:

Setting up Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


It is highly recommended to create a new, separate environment for FM2PROF. To create a new environment, open an (Anaconda) prompt and navigate to the directory of the python source code, and type: 

.. code-block:: bash

	conda env create -f environment.yml

This will create a new Python 3.6 environment called "fm2prof", with necessary packages installed. Next, install the BRL package and its dependencies. Open a terminal and navigate to the BRL directory that contains *setup.py*. First, activate the environment:

.. code-block:: bash

	conda activate fm2prof


Install the python package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Then, install the package:

.. code-block:: bash

	conda develop .



.. note::
    We're using ``conda develop`` because the code may change a lot during development. This ensures that all changes to the code are immediately available within your conda environment. 