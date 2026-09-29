# How to set up a 2D simulation

## Simulation & boundary conditions

To produce sensible cross-sections, the 2D simulation must be run under specific conditions:

1. **Water levels should rise monotonically** at every point in the system. We recommend an exponentially rising discharge boundary upstream, combined with a linearly rising water level boundary downstream. Several iterations may be needed to tune the boundary conditions and achieve satisfactory results.
2. **FM2PROF does not extrapolate elevations above the highest water level.** Make sure your boundary conditions cover the full range of water levels expected in the 1D model's use case.
3. **Elevations below the lowest water level are accounted for**, but this part of the cross-section is extrapolated based on certain assumptions. See [water level independent geometry](tech_docs/glossary.md#water-level-independent-geometry) for details.

## Output settings

The 2D model output must include:

- **Mesh geometry**: face coordinates, bed levels, and flow areas
- **Hydraulic results**: water levels, water depths, velocities, and Chézy roughness

Enable these in the 2D model's configuration file.

Keep the number of output timesteps ("maps") reasonable — too many will noticeably slow down FM2PROF. Aim for a balance between the number of maps and the accuracy of the resulting cross-section.
