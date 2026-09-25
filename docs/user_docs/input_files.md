# Input Files

FM2PROF requires two mandatory input files and accepts two optional files.

| File | Required |
|------|----------|
| 2D Data | Yes |
| Cross-section locations | Yes |
| Region polygon | No |
| Section polygon | No |



## 2D Data

FM2PROF supports two forms of 2D data: hydraulic data or a DEM (elevation only). If a DEM is provided, FM2PROF cannot generate roughness fields, nor distinguish between storage (dead zones) and conveyance, since this requires hydraulic simulation results. Configured via `2DMapOutput` in the ini file.

### Hydraulic data

FM2PROF currently only supports **NetCDF (`*_map.nc`)** output from a D-Flow FM 2D simulation. The 2D simulation must be set up in a specific way to produce sensible cross-sections — see [how to prepare a 2D simulation](2d_simulation.md) for details.

### DEM (elevation only)

Two DEM input formats are supported: CSV and GeoTIFF. See [Cross-sections from geometry alone](elevation_only.md) for details on preparing input and interpreting the generated output.

#### CSV

A comma-separated file with a header row. The following columns are required:

| Column | Description |
|--------|-------------|
| `POINT_X` | Face centroid x-coordinate |
| `POINT_Y` | Face centroid y-coordinate |
| `POINT_Z` | Bed level |
| `Shape_Area` | Face area |

A CSV source contains geometry only — no hydraulic results are available, which affects the resulting output. 

#### GeoTIFF

A `*.geotif` raster file, following common GeoTIFF conventions.

## Cross-section locations

A file defining the 1D cross-section locations onto which the 2D data is projected. Each record specifies a unique identifier, representative length, branch identifier, chainage, and (x, y) coordinates.

Configured via `CssFile` in the ini file.



## Region polygon file (optional)

A GeoJSON polygon file that divides the model domain into named [regions](../tech_docs/glossary.md#regions). It should be a valid [MultiPolygon geojson file](api.md#fm2prof.polygon_file.MultiPolygon). 

When this file is provided, each region is processed independently during the nearest-neighbour classification step. As a result, cross-sections contained within one region do not see or use 2D points contained within another region.

If omitted, all 2D faces are classified to the nearest cross-section without regional constraints.

Configured via `RegionPolygonFile` in the configuration file. See [technical manual](../tech_docs/conceptual_design.md#region-polygon-file) for how it is used during runtime. 



## Section polygon (optional)

A GeoJSON polygon file that assigns 2D faces and edges to roughness [sections](../tech_docs/glossary.md#sections)] (e.g. main channel, floodplain). Labels defined in this file are used directly as section identifiers in the output roughness files. It should be a valid [MultiPolygon geojson file](api.md#fm2prof.polygon_file.MultiPolygon).

When this file is provided, it is used to determine which parts of the cross-section are assigned to which roughness class. 

If omitted, but hydraulic data is provided, FM2PROF will use a [variance reduction method](https://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction) to make a distinction between alluvial- and non-alluvial roughness fields. 

Configured via `SectionPolygonFile` in the configuration file. See [technical manual](../tech_docs/conceptual_design.md#section-polygon-file) for how it is used during runtime. 
