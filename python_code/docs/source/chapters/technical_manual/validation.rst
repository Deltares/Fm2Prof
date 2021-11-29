Validation cases
====================

This section presents seven idealised test cases. Every idealised case is used to validate a different aspect for river geometry. The test cases are ordered from simple to more complex. All idealised cases are straight channels of 3000m length. The slope is always 3.33e-4, making the total drop in elevation over the whole length 1m. The channel width is 150m, and if a floodplain is present, it is 50m wide:

.. figure:: ../figures/tests_setup.png
	
	Diagram of validation cases


The only exception to this is case 7 (triangular grid), which has larger dimensions but still follows the same channel design. Every test is ran using a monotonically rising boundary condition upstream and a stage-discharge relation downstream. 

01 - Rectangular cross-section
---------------------------------

Description
..............

The first case is a uniform rectangular channel. There is no main channel or floodplain, and
only one roughness value has been used. The cross-section shape is as follows:

.. figure:: ../figures/test01_crosssection.PNG
	:alt: cross-section for case 1.

	Cross-section for case 1.

The case is meant to test for the following:

- Construction of the cross-section profile for simple geometry.
- Reconstruction of one roughness curve.
- Reconstruction of one roughness section width.


Results
..............

.. image:: ../../../../tests/Output/RunWithFiles_Output/case_01_rectangle/CaseName01/figures/case1_0.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_01_rectangle/CaseName01/figures/case1_500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_01_rectangle/CaseName01/figures/case1_1000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_01_rectangle/CaseName01/figures/case1_1500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_01_rectangle/CaseName01/figures/case1_2000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_01_rectangle/CaseName01/figures/case1_2500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_01_rectangle/CaseName01/figures/case1_3000.png 

02 - Compound cross-section
---------------------------------
Description
..............
The second case is a uniform compound channel. The main channel and floodplain have their
own roughness values and the geometry is more complex when compared to the first case.

.. figure:: ../figures/test02_crosssection.PNG
	:alt: cross-section for case 2.

	Cross-section for case 2.

The case is meant to test for the following:

- Construction of the cross-section profile for more complex geometry, specifically the transition from main channel to floodplain.
- Reconstruction of the main channel and floodplain section widths.
- Reconstruction of two roughness curves.

Results
..............
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_02_compound/CaseName01/figures/case2_0.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_02_compound/CaseName01/figures/case2_500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_02_compound/CaseName01/figures/case2_1000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_02_compound/CaseName01/figures/case2_1500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_02_compound/CaseName01/figures/case2_2000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_02_compound/CaseName01/figures/case2_2500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_02_compound/CaseName01/figures/case2_3000.png 

03 - Threestage cross-section
---------------------------------

Description
..............
The third case is a more complex version of the compound channel, namely a three-stage
compound channel. The main channel has the same dimensions, but the floodplain is now
separated into two parts with a different elevation.

.. figure:: ../figures/test03_crosssection.PNG
	:alt: cross-section for case 3.

	Cross-section for case 3.

The case is meant to test for the following:

- Construction of the cross-section profile for more complex geometry, specifically more detailed floodplains.
- Reconstruction of two roughness curves from non-homogeneous floodplain.

It is expected that the roughness values for the floodplain will be an average of the two
original curves (one for each stage in the floodplain).

Results
..............
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_03_threestage/CaseName01/figures/case3_0.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_03_threestage/CaseName01/figures/case3_500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_03_threestage/CaseName01/figures/case3_1000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_03_threestage/CaseName01/figures/case3_1500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_03_threestage/CaseName01/figures/case3_2000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_03_threestage/CaseName01/figures/case3_2500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_03_threestage/CaseName01/figures/case3_3000.png 

04 - Storage
---------------------------------
Description
..............

The fourth case is meant to check whether storage is correctly identified by FM2PROF. The
topography is the same in case 2: a compound channel. Thin dams have been added to the
floodplain in a control volume to create a storage area. Thin dams are infinitely thin and infinitely high walls that can be added to Flexible Mesh models. By adding thin dams to each cell in the perpendicular direction of the flow, water can enter the cell but doesn’t flow to the neighbouring cells, effectively creating a dead zone or storage area. 

.. figure:: ../figures/test04_crosssection.PNG
	:alt: cross-section for case 4.

	Cross-section for case 4.


The case is meant to test for the following:

- Construction of flow cross-sections, and by extension...
- the generation of storage sections in the total cross-section (total cross-section minus flow cross-section gives storage area in SOBEK).

It is expected that the storage will be slightly underestimated due to velocities in cells near
the main channel being higher (and therefore possibly classified as not storage). Furthermore,
the waterlevels upstream of the thin dams are expected to be underestimated in the 1D model,
due to the build up of water that is captured in the 2D model but which is not present in the
1D model because only storage is added, not a barrier such as a thin dam.

Results
..............
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_04_storage/CaseName01/figures/case4_0.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_04_storage/CaseName01/figures/case4_500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_04_storage/CaseName01/figures/case4_1000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_04_storage/CaseName01/figures/case4_1500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_04_storage/CaseName01/figures/case4_2000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_04_storage/CaseName01/figures/case4_2500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_04_storage/CaseName01/figures/case4_3000.png 

.. _validation_summerdike:

05 - Summer dike
---------------------------------
Description
..............

Floodplains that contain structures that initially obstruct flow, but overflow at higher discharges, (i.e. compartimentalized floodplains) create a distinct effect in available volume that cannot be reproduced using a regular one-dimensional cross-section. Embankment along the main channel are an example of such a feature. 
SOBEK has a volume correction function that is designed to simulate this behaviour. This testcase is designed to test this behaviour. The embankments are added along the main channel and are 1m high.

.. figure:: ../figures/test05_crosssection.PNG
	:alt: cross-section for case 5.

	Cross-section for case 5.


The case is meant to test for the following:

- A correct adjustment of the volume-waterlevel curve.

Results
..............
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_05_dyke/CaseName01/figures/case5_0.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_05_dyke/CaseName01/figures/case5_500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_05_dyke/CaseName01/figures/case5_1000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_05_dyke/CaseName01/figures/case5_1500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_05_dyke/CaseName01/figures/case5_2000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_05_dyke/CaseName01/figures/case5_2500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_05_dyke/CaseName01/figures/case5_3000.png 

06 - Lakes
---------------------------------
Description
..............

Lakes or ponds are water bodies in the floodplain that are not part of the main channel and should therefore be ignored until they start to contribute to the flow. For this test case with the lake the compound channel topography was taken as the basis. The 2D mesh was made finer to allow the addition of a pond to the floodplains. The pond was
added to the middle control volume on one of the floodplain banks (between 1250m and 1750m). The pond is 10m deep.

.. figure:: ../figures/test06_crosssection.PNG
	:alt: cross-section for case 6.

	Cross-section for case 6.


The case is meant to test for the following:

- Construction of the cross-section profile, specifically identifying the pond and masking it out from the cross-section generation until the pond is flooded.


Results
..............
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_06_plassen/CaseName01/figures/case6_0.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_06_plassen/CaseName01/figures/case6_500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_06_plassen/CaseName01/figures/case6_1000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_06_plassen/CaseName01/figures/case6_1500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_06_plassen/CaseName01/figures/case6_2000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_06_plassen/CaseName01/figures/case6_2500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_06_plassen/CaseName01/figures/case6_3000.png 

07 - Triangular grid
---------------------------------
Description
..............

Results
..............
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_07_triangular/CaseName01/figures/case7_0.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_07_triangular/CaseName01/figures/case7_1200.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_07_triangular/CaseName01/figures/case7_2400.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_07_triangular/CaseName01/figures/case7_3600.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_07_triangular/CaseName01/figures/case7_4800.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_07_triangular/CaseName01/figures/case7_6000.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_07_triangular/CaseName01/figures/case7_7200.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_07_triangular/CaseName01/figures/case7_8500.png 
.. image:: ../../../../tests/Output/RunWithFiles_Output/case_07_triangular/CaseName01/figures/case7_10000.png 