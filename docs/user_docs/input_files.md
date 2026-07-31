# Input Files

FM2PROF requires two mandatory input files and accepts two optional files.

| File | Required |
|------|----------|
| 2D map output | Yes |
| Cross-section locations | Yes |
| Region polygon | No |
| Section polygon | No |



## 2D map output

The 2D model output file. Two formats are currently supported:

**NetCDF (`*_map.nc`)** — output from a D-Flow FM 2D simulation. Contains mesh geometry (face coordinates, bed levels, flow areas) and hydraulic results (water levels, water depths, velocities, Chézy roughness). 

**CSV** — a comma-separated file with a header row. The following columns are required:

| Column | Description |
|--------|-------------|
| `POINT_X` | Face centroid x-coordinate |
| `POINT_Y` | Face centroid y-coordinate |
| `POINT_Z` | Bed level |
| `Shape_Area` | Face area |

A CSV source contains geometry only; no hydraulic results are available. This will impact the output. See [Cross-sections from geometry alone](elevation_only.md) for more information on creating input and the generated output. 

Configured via `2DMapOutput` in the ini file.


## Cross-section locations

A file defining the 1D cross-section locations onto which the 2D data is projected. Each record specifies a unique identifier, representative length, branch identifier, chainage, and (x, y) coordinates.

Configured via `CssFile` in the ini file.



## Region polygon (optional)

A GeoJSON polygon file that divides the model domain into named regions. Each region is processed independently during the nearest-neighbour classification step, so that cross-sections in one region do not attract 2D points from another.

If omitted, all 2D faces are classified to the nearest cross-section without regional constraints.

Configured via `RegionPolygonFile` in the ini file.



## Section polygon (optional)

A GeoJSON polygon file that assigns 2D faces and edges to roughness sections (e.g. main channel, floodplain). Labels defined in this file are used directly as section identifiers in the output roughness files.

If omitted, section classification is derived automatically from the spatial variance of the Chézy roughness field.

Configured via `SectionPolygonFile` in the ini