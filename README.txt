* Documentation *

For a full documentation, installion instructions and how-to see /doc.  

* Environment: *
A file containing the virtual environment (environment.yml) dependencies is provided and strongly recommended to use in order to execute successfully FM2PROF.
To install it simply run the following:
conda env create -f environment.yml
For more information:
[conda environment] - https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

* About the packages: *
This package contains a dependency to GEOS (shapely) which needs to be present in your system as it's a C library.
You can install this directly with conda when running on a virtual environment with the following line:
conda install -c anaconda geos
For more information:
[shapely] - https://pypi.org/project/Shapely/
[GEOS] - https://anaconda.org/anaconda/geos