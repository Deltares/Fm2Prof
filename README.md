# FM2PROF

[![ci](https://github.com/Deltares/fm2prof/actions/workflows/ci.yml/badge.svg)](https://github.com/Deltares/fm2prof/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Deltares_Fm2Prof&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Deltares_Fm2Prof)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Deltares/fm2prof)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Deltares/fm2prof)
[![Available on pypi](https://img.shields.io/pypi/v/fm2prof.svg)](https://pypi.python.org/pypi/fm2prof)
[![Formatting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)

## What is FM2PROF?

FM2PROF (Flow**FM** **to** **Prof**ile) is a Python package that extracts 1D cross-sectional data from 2D hydrodynamic model results. It bridges the gap between detailed 2D flood modelling and efficient 1D hydraulic analysis by automatically generating cross-section geometries and roughness parameters from 2D FlowFM simulation outputs.

### How It Works

1. **Input**: 2D FlowFM map files (NetCDF) + cross-section location definitions
2. **Processing**: Spatial classification, geometry extraction, roughness calculation
3. **Output**: 1D model files compatible with 1D hydraulic software (currently supported: D-Flow 1D, SOBEK 3)

```
2D FlowFM Results    →    FM2PROF    →    1D Cross-Sections
(detailed mesh)           (extraction)    (efficient profiles)
```

## Quick Start

### Installation

**Option 1: Python Package (Recommended)**
```bash
pip install fm2prof
```

**Option 2: Windows Executable**
Download from [GitHub Releases](https://github.com/Deltares/Fm2Prof/releases)

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
FM2PROF create MyProject

# Edit MyProject.ini with your file paths, then run
FM2PROF run MyProject --overwrite
```

### Required Input Files

| File Type | Description | Format |
|-----------|-------------|---------|
| **2D Map Output** | 2D simulation results | NetCDF (.nc) |
| **Cross-Section Locations** | Where to extract profiles | CSV/TXT with X,Y,Branch,Chainage |

**Optional:**
- **Region Polygons**: Define floodplain vs main channel areas (GeoJSON)
- **Section Polygons**: Specify extraction boundaries (GeoJSON)

### Example Output

FM2PROF generates:
- **Cross-section geometries** (bed levels, widths, areas)
- **Roughness tables** (Manning's n values per sub-section)
- **Visualisation plots** for quality checking

## Documentation & Examples

- 📚 **Full Documentation**: [deltares.github.io/Fm2Prof](https://deltares.github.io/Fm2Prof/)
- 🚀 **Tutorial**: [Getting Started Guide](https://deltares.github.io/Fm2Prof/markdown/quickstart/)
- 📓 **Jupyter Notebooks**: Interactive examples in `/notebooks`
- 🔧 **Configuration Reference**: Complete parameter documentation

## Key Features

### 🌊 **Hydraulic Intelligence**
- Volume-preserving 1D geometry
- Roughness weighting methods (area-based, distance-based)
- Flood-dependent storage areas
- Transition height calculations for overbank flow

### ⚡ **Performant**
- Leverages MeshKernel for fast polygon operations
- Efficient NetCDF data handling
- K-Nearest Neighbour for rapid classification

### 🔧 **Flexible Configuration**
- INI-based configuration files
- Python API for programmatic control

### 📊 **Multiple Output Formats**
- D-Flow 1D compatible files
- SOBEK 3 format support
- Generic CSV/JSON outputs
- Visualisation and diagnostic plots

## Use Cases

### Typical Workflow

1. **Run 2D FlowFM Model**: Generate detailed flood simulation
2. **Define Cross-Sections**: Specify where 1D profiles are needed
3. **Configure FM2PROF**: Set extraction parameters and output options
4. **Extract Profiles**: Run FM2PROF to generate cross-sectional data
5. **Build 1D Model**: Import results into 1D hydraulic software
6. **Validate**: Compare 1D vs 2D results for key locations

### Real-World Applications

- **Dutch National Flood Forecasting River Models**: all river models are using `fm2prof` generated profiles and roughnesses 
- **Maeslant barrier forecasting model**: the hydraulic forecasting of the Maeslant Barrier uses `fm2prof` generated geometry and roughnesses

## Requirements

- **Python**: 3.10 or higher
- **Key Dependencies**: NumPy, Pandas, NetCDF4, Shapely, [MeshKernel](https://github.com/Deltares/MeshKernelPy)
- **Supported 2D software**: D-FlowFM, D-Hydro
- **Supported 1D Software**: D-Flow 1D, SOBEK 3

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for:
- 🐛 Bug reports and feature requests
- 💻 Code contributions and pull requests  
- 📖 Documentation improvements
- 🧪 Test case submissions

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

**Ready to extract 1D profiles from your 2D model?** 
Start with our [Quick Tutorial](https://deltares.github.io/Fm2Prof/markdown/quickstart/) or explore the [example notebooks](notebooks/) to see FM2PROF in action.
