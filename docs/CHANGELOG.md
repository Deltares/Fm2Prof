## v2.4.0 (05-09-2025)

This update introduces a new built-in method to use region & section polygons 
built on the MeshKernel package, switches to the uv package and project
manager and increases test coverage. 

### New features

- new built-in method to use region- and section polygons, enabled by default, see [documentation](https://deltares.github.io/fm2prof/markdown/conceptual_design/#region-polygon-file) for more information
- missing key in configuration now prints a warning
- new configuration parameters "defaultsection" and "defaultregion"
- configuration parameter "classificationmethod" no longer exists

### Breaking changes

- old "DeltaShell" classification with undocumented "_BATHY.nc" no longer works
- Region polygon now requires "region" property as well as "name" property.
- Section polygon now requires the "name" property as well as the "section" property

### Maintenance

- increased test coverage
- removed archived tests


## v2.3.3 (24-09-2024)

### New features

- added additional statistics: last3, max3 (#92)
- all statistics are now written to csv file (#92)

### Bugs fixed
- fm2prof interpreting missing value (-999) as water level (#89 )
- fixed branch exception issue that failed if user provided no rules in `utils.GenerateCrossSectionLocationFile`

### Maintenance
- switched from black & isort to Ruff for linting and code formatting
- removed unused code
- added extra tests

## v2.3.2 (07-05-2024)

This release focuses on improving the `fm2prof.utils` tooling

### New functionality

- `utils.Compare1D2D` has new option to output a `last25` and `max13` longitudinal view, and to combine the output of several simulations into a single figure (#65). The docs have been updated with a notebook to showcase this new functionality. See User Manual -> Utilities

### Documentation

- added notebooks to documentation that describe the use of some utilities


### Changes

- added new `sito_2024` `PlotStyle` that is now default for `Compare1D2D` (#66)

### Deprecations

- `Compare1D2D.figure_longitudinal_time` is now deprecated in favour of `Compare1D2D.figure_longitudinal` with `stat="time"` parameter



## v2.3.1 (24-04-2024)

**Bug fixes**
- Fixed a bug that threw an exception if user did not specify a stop time when using `utils.Compare1D2D` (#78)
- Fixed bug (#81) that prevented executable from being build because of missing favico

**Documentation**
- added python snippets to quickstart tutorial


## v2.3.0 (19-04-2024)

### New functionality

- Configuration file now has a new `debug` section with debug specific parameters. This section includes two parameters previously in the general `parameters` section (`ExportMapFiles` and `CssSelection`) and the new `ExportCSSData`. 
- new debug option `ExportCSSData` that when enabled output data to analyse cross-section generation. 
- new option `ConveyanceDetectionMethod` to toggle between the previous way to detect storage (0) and a the new one (1, default). 

### Documentation

- documentation now includes a notebook specifying how output from `ExportCSSData` option can be used to analyse flow data
- docstrings of cross-section class updated to describe flow/storage separation methodology
- several chapters ported over from sphinx

### bug fixes & chores
- added dedicated tests for cross-section class
- test coverage is now reported in Sonarcloud
- fixed bug that caused error while writing log

### bug fixes & chores

- fixed bug in `utils` that threw an error when using matplotlib 3.7 or higher
- removed unused code blocks
- updated type hinting and code documentation of `CrossSection.py`



## v2.2.8 (2023-10-03)

This version update FM2PROF to Python 3.10 or higher. It removes unused dependencies and updates the package system `poetry` to version 1.8.2. Due to this switch, the commitizen workflow is currently not used, and the documentation system switch from sphinx to mkdocs. 

This is the first version to be published to [PyPi](https://pypi.org/project/fm2prof/2.2.8/), which means that FM2PROF can now be installed using pip

`pip install fm2prof`

Known issues:

- The executable is not available for this version. 

## v2.2.7 (2023-10-03)

This version adds the parameter stoptime to utils.Compare1D2D and makes both starttime and stoptime parameters optional. With these parameters users can crop the section of the results over which statistics will be computed and figures made.

## v2.2.6 (2023-10-01)

- Fixed an issue where irregular station names like `MA_67.00Z` caused a sorting error in `utils.Compare1D2D`

## v2.2.5 (2023-07-28)

- FM2PROF now validates the "SkipMaps" parameter and throws error if its value is larger than the available number of maps
- Fixed bug that threw exception if roughness tables could not be produced during finalization step
- Fixed bug that prevented BranchRules file to be read if multiple empty lines existed at the end of the file
- Fixed bug where `utils.Compare1D2D` would throw exception if input netCDF files did not exist, but csv files did. 
- Fixed bug in `utils.Compare1D2D` where execution failed if a QH relationship could not be produced
- Statistics are no longer computed on initialization of `utils.Compare1D2D`. Instead, they are not computed when requested during evaluation. 

## v2.2.4 (2023-07-05)

- Implemented 'onlyFirst' and 'onlyLast' rules for BranchRules file

## v2.2.3 (2022-12-21)

- Implemented functionality to compare two 1D models for bed level change, Summer dike change and width change

## v2.2.2 (2022-08-26)

### Fix

- wrong method output type
- catching wrong exception
- exception in utils

## v2.2.1 (2022-08-24)

## v2.2.0 (2022-07-13)

### Fix

- log style now the same as stream, added support for tqdm
- added cross-section progress to log
- revised logger style (#34)
- first figure does not use correct style
- added 10 cm tolerance to section width correction
- main section width check
- output path compare1d2d discharge figure
- issue 33

## v2.1.2 (2022-07-05)

## v2.1.1 (2022-07-04)

### Fix

- missing index.rst

## v2.1.0 (2022-07-04)

### Fix

- macos/linux posixpath fail fix
- isolated set_locale and wrapped in try/except
- possibly fix posix path error with trailing whitespace
- run with ini suffix fix (#29)
- docs now only build on master

### Feat

- expanded cli
- groundwork for expansion of cli (#31, #30)
- overwrite option for output, single output folder (#31)
- added new tools to utils

## v2.0.0 (2022-06-28)

### Fix

- **IniFile**: fixed bug introduced by switching to pathlib
- output path now relative to config file

## v1.5.3 (2022-05-27)

### Fix

- adding cm accuracy to section width check (#23)

## v1.5.2 (2022-05-27)

### Fix

- adding cm accuracy to section width check (#23)
- main section width check
- fixed bug introduced by earlier fix :p
- files now relative to config file (#24)

## v1.5.1 (2022-05-26)

### Fix

- **CrossSection.py**: add main section width requirement (#23)

## v1.5.0 (2022-05-26)

### Fix

- **CrossSection.py**: add main section width requirement (#23)
- sc bug

### Feat

- **cli**: new cli with poetry script hook & python -m

## v1.4.4 (2022-05-03)

### Fix

- update black to 22.3 (#16)

## v1.4.3 (2021-11-29)
