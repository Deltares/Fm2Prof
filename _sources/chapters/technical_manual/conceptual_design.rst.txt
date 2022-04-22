Conceptual design
====================

This chapter describes in more detail the steps FM2PROF takes to go from a 2D representation of reality to a 1D representation of reality. Conceptually, FM2PROF works in three distinct steps: (1) initialisation, (2) building cross-sections and (3) finalisation. 

Initialisation
--------------------

.. thumbnail:: ../figures/conceptual_initialisation.png
            :align: center
            :alt: alternate text
            :width: 50%

The initialisation step involves parsing (i.e. reading and making available for further analysis) the 2D data. In this step control volumes and sections are defined as well. This step may take some time to complete, but this preprocessing greatly reduces the computation times in the next step. 


Import region file 
........................

The :term:`Region` file is specified in the configuration file. 

.. warning::
    For any reasonably sized river model, it is currently not advised to supply a polygon. Instead, a NetCDF file should be supplied. See :ref:`issue_region_polygon` for more information. 

Import section file 
........................

The :term:`Section` file is specified in the configuration file. 

.. warning::
    For any reasonably sized river model, it is currently not advised to supply a polygon. Instead, a NetCDF file should be supplied. See :ref:`issue_region_polygon` for more information. 


Import 2D data
........................

.. _section_parsing_2d_data:

Parsing 2D data
,,,,,,,,,,,,,,,,,,

Dflow2d uses a staggered grid to solve the (hydrostatic) flow equations. Because of this staggered approach, there is not a single 2D point that has all information. Flow information (flow velocity, discharge) is stored in `flow links`, while geometry (bed level) is stored in cell faces. |project| needs both information from the faces, as from the links. 

.. thumbnail:: ../figures/dflow2d_grid.PNG
            :align: center
            :alt: alternate text
            :width: 50%

            The dflow2d staggered grid. 

Below is a table that lists all variables read by |project| from dflowd map output. 

.. csv-table:: Overview of variables from the 2D model that are used by FM2PROF
   :file: ../tables/dflow2d_keys.csv
   :widths: 50, 50
   :header-rows: 1


Classification of volumes
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

:term:`Control volumes` are used to define which 2D datepoints are linked to which 1D cross-section. This is done in the following steps:

- Each node, link and face is assigned a :term:`Region`. This is currently done through DeltaShell (see :ref:`issue_region_polygon`)
- For each region seperately, a scikit-learn `KNearestNeighbour <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_ classifier is trained. 
- The classifier is used to uniquely identify each 2D link and each face to a 1D cross-section

Classification of sections
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

:term:`Sections <Section>` are used to output a different roughness function for the main channel and the floodplains. The purpose of the classification is to determine whether a 2D point belongs to the main channel section, or to the floodplain section (see warning below). 

Two methods are implemented:

- :ref:`Variance based classification<section_classification_variance>`
- Polygon-based classification using DeltaShell (see :ref:`issue_region_polygon`)


.. warning:: 
    If classification is done using DeltaShell, it is possible to define more than two sections. However, this functionality is not tested and may not work properly. 


Build Cross-Section
--------------------

.. thumbnail:: ../figures/conceptual_build.png
            :align: center
            :alt: alternate text
            :width: 50%

Once initialisation is complete, |project| will loop over each :term:`cross-section location<Cross-section location>`. In each iteration, the program takes two steps: (1) building the geometry and (2) building the roughness tables. 

.. warning::
    No cross-section will be generated for locations that have no 2D data assigned or have less than 10 faces assigned. This may happen if a location lies outside the 2D grid, or if there are many cross-section closely together. If this happens, an error is raised by FM2PROF. The user should check the cross-section location input file to resolve the problem. 

Build Geometry
........................

In each loop, a number of steps is taken based on the 2D data that is uniquely assigned to that cross-section:

- :term:`Lakes` are identified using the :ref:`identify_lakes`
- :term:`Flow volume` and :term:`Storage volume` are seperated using the :ref:`distinguish_storage`
- The :term:`water level dependent geometry<Water level (in)dependent geometry>` is computed using :ref:`wl_dependent_css`
- The :term:`water level independent geometry<Water level (in)dependent geometry>` is computed using :ref:`wl_independent_css`
- The parameters for :term:`Summerdikes` are defined using the :ref:`sd_optimisation`
- Finally, the cross-section is simplified using the Visvalingam-Whyatt method of poly-line vertex reduction :ref:`simplify_css`

Build roughness
........................

At each cross-section point, a roughness look-up table is constructed that relates water level (in m + NAP) to a Chézy roughness coefficient. This is done in three steps:

- For each section, a roughness table is constructed by averaging the 2D points 




Finalisation
--------------------

.. thumbnail:: ../figures/conceptual_finalisation.png
            :align: center
            :alt: alternate text
            :width: 50%


