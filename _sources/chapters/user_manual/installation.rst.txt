.. _ChapterInstallation:


Installation Instructions
=========================

There are several ways to build or install |project|, depending on your use case. 


Installation from PyPI packages
---------------------------------

PyPi is currently not supported


Installation from local checkout
--------------------------------

There are several ways to install |project| to your local python virtual environment. 

Using poetry
^^^^^^^^^^^^

Activate a python 3.7 virtual environemnt. Next, checkout |project| from the git repository:

.. code-block:: shell

	git clone https://github.com/deltares/fm2prof
	cd fm2prof
	poetry install

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


Recommended other software
---------------------------------

Minimum requirements
^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    Some requirements are available from multiple sources and distributions. We recommend using `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage the python environment 

* `Python <http://www.python.org/>`_ ** == 3.6**. The conda distribution is highly recommended. Python 2 syntax is not supported. The software is developed on 3.6, but earlier versions may work as well. 


Optional Requirements
^^^^^^^^^^^^^^^^^^^^^^^^

* **QGIS** to visualise FM2PROF output
