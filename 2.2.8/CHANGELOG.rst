## v2.2.7 (2023-10-03)

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
- (#34)
- revised logger style (#34)
- first figure does not use correct style
- added 10 cm tolerance to section width correction
- main section width check
- output path compare1d2d discharge figure
- #33

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
