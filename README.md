# FM2PROF

<!-- Project information -->
[![Python 3.11+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://deltares.github.io/Fm2Prof/)
[![Formatting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)

<!-- QA -->
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Deltares_Fm2Prof&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Deltares_Fm2Prof)
[![ci](https://github.com/Deltares/fm2prof/actions/workflows/ci.yml/badge.svg)](https://github.com/Deltares/fm2prof/actions/workflows/ci.yml)

<!-- Release -->
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Deltares/fm2prof)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Deltares/fm2prof)
[![Available on pypi](https://img.shields.io/pypi/v/fm2prof.svg)](https://pypi.python.org/pypi/fm2prof)

## What is FM2PROF?

FM2PROF is a Python package to build 1D flow profiles (cross-sections) from 2D flexible meshes. It bridges the gap between detailed 2D flood modelling and efficient 1D hydraulic analysis by automatically generating cross-section geometries and roughness parameters from 2D FlowFM simulation outputs.

```
2D Mesh            →    FM2PROF    →    1D Cross-Sections
(detailed mesh)       (extraction)         (profiles)
```

"Flexible Mesh" means that the 2D input does not presume a structured grid - points can be scattered over the map. Currently supported input formats are: 

- Delft3D Flexible Mesh netcdf map files
- D-HYDRO netcdf map files
- a CSV file with x,y,z and area information 

For detailed information on the format of input files see the [documentation](https://deltares.github.io/Fm2Prof/latest/user_docs/input_files). 

FM2PROF currently support the following output format for 1D files:

- SOBEK 3 cross-sections and roughness files
- D-HYDRO 1D cross-sections and roughness files


## Quick Start

### Installation

```bash
pip install fm2prof
```

### Basic Usage

**Python API:**
```python
from fm2prof import Project

# Load configuration and run
project = Project('config.ini')
project.run()

# Or configure programmatically
project = Project()
project.set_input_file('2DMapOutput', 'model_map.nc')
project.set_input_file('CrossSectionLocationFile', 'crosssections.csv')
project.set_output_directory('./output')
project.run()
```

**Command Line:**
```bash
# Create new project
uv run python fm2prof create MyProject

# Edit MyProject.ini with your file paths, then run
uv run python fm2prof run MyProject --overwrite
```

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/Deltares/Fm2Prof/issues)
- **Contact**: FM2PROF development team at Deltares
- **License**: GPL-3.0-or-later AND LGPL-3.0-or-later

## Citation

If you use FM2PROF in research, please cite:
```
FM2PROF Development Team (2024). FM2PROF: FlowFM to Profile Extraction Tool. 
Deltares. https://github.com/Deltares/Fm2Prof
```

---
