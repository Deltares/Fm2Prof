# FM2PROF

[![ci](https://github.com/Deltares/fm2prof/actions/workflows/ci.yml/badge.svg)](https://github.com/Deltares/fm2prof/actions/workflows/ci.yml)
[![docs](https://github.com/Deltares/fm2prof/actions/workflows/docs.yml/badge.svg)](https://github.com/Deltares/fm2prof/actions/workflows/docs.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Deltares_Fm2Prof&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Deltares_Fm2Prof)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Deltares/fm2prof)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Deltares/fm2prof)

Python package to create 1D river models from 2D river models. 


## Introduction

FM2PROF is a Python package to help create 1D models from 2D models. It's primarily built for river systems, but has been applied to lakes as well. 

The main purpose of FM2PROF is to generate 1D cross-sections and roughness tables using information from a 2D model. Users should provide 




## Supported software

FM2PROF currently works with the following model software systems and versions

| Type | Supported | 
| --- | --- | 
| 1D models | SOBEK 3.6+ | 
| 2D model | Delft3D Flexible Mesh 2019+ | 

## Getting started &  Documentation.
You can find the documentation here: [https://deltares.github.io/fm2prof/](https://deltares.github.io/fm2prof/).

## Development

- We enforce [black](https://github.com/psf/black) formatting on push. So after pushing, pull to get the reformatted code, or do a black reformat locally. 
- Direct commits to master are not allowed (except for admins)
- We use [commitizen tools](https://commitizen-tools.github.io/commitizen/bump/) to bump version. Version bump is done manually after a pull request. E.g. after a minor improvement:

```bash
# first do a dry run to test
poetry run cz bump --increment MINOR --dry run
# then for real
poetry run cz bump --increment MINOR
```


