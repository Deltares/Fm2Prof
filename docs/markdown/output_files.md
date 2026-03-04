# Output Files

FM2PROF generates a set of output files that can be used to set up 1D hydraulic models. This page describes the output files that are generated and their purpose.

## Output Directory Structure

After running FM2PROF, the output files are organized in subdirectories:

```
output/
├── fm2prof.log 
├── dflow1d/          # D-Flow 1D format files
│   ├── CrossSectionLocations.ini
│   ├── CrossSectionDefinitions.ini
│   ├── roughness-Main.ini
│   ├── roughness-FloodPlain1.ini
│   ├── roughness-FloodPlain2.ini
│   └── volumes.csv
└── dhydro/           # D-Hydro (1D) format files 
│   ├── crsloc.ini
│   ├── crsdef.ini
│   ├── roughness-Main.ini
│   ├── roughness-FloodPlain1.ini
│   ├── roughness-FloodPlain2.ini
│   └── volumes.csv
└── debug/           # Optional debug files (if enabled in configuration)
    ├── cross_section_volumes.geojson
    ├── face_output.geojson
    └── edge_output.geojson
```


!!! note
    The "dflow1d" format is only used by the SOBEK 3 software from version 3.5
    and higher. Support for SOBEK 3.4 and lower has been dropped since FM2PROF
    version 2.5





### File Differences between dhydro and dflow1d (SOBEK 3)
The `dhydro` and `dflow1d` formats are alternative output formats with slightly different file naming conventions. The `dflow1d` is used by SOBEK 3, while `dhydro` is used by the D-Hydro 1D2D software. 

| `dflow1d` | `dhydro` | Description |
|-----------|--------------|-------------|
| CrossSectionLocations.ini | crsloc.ini | Cross-section locations |
| CrossSectionDefinitions.ini | crsdef.ini | Cross-section definitions |
| roughness-*.ini | roughness-*.ini | Roughness files (same naming) |
| volumes.csv | volumes.csv | Volume comparison (same naming) |


## Output Format

The D-Flow 1D format is the default output format and generates files compatible with Deltares D-Flow 1D software.

### CrossSectionLocations.ini / crsloc.ini

This file contains the spatial locations of the cross-sections along the river or channel network.

**Example:**
=== "dflow1d"

    ```ini
    [CrossSection]
        id					= channel1_125.000
        branchid			= Channel1
        chainage			= 125.0
        shift				= 0.000
        definition			= channel1_125.000

    ```
=== "dhydro"

    ```ini
    [CrossSection]
        Id					= channel1_125.000
        branchId			= Channel1
        chainage			= 125.0
        shift				= 0.000
        definitionId		= channel1_125.000

    ```

### CrossSectionDefinitions.ini / crsdef.ini

This file contains the geometrical definitions of each cross-section, including the shape and dimensions.

**Example**

=== "dflow1d"

    ```ini
    [Definition]
	id					= channel1_125.000
	type				= tabulated
	thalweg				= 0.000
	numLevels			= 20
	levels				= -2.0249 -1.9486 -1.9476 -1.8829 -1.8261 -1.7498 -1.6281 -1.5567 -1.4315 -1.3622 -0.5710 -0.0747 -0.0391 -0.0216 -0.0049 0.0695 0.4215 0.8350 1.2430 1.7807
	flowWidths			= 50.0000 50.0090 50.0100 50.0150 50.0210 50.0250 50.0330 50.0370 50.0450 50.0490 50.0980 50.1290 50.1310 50.1320 150.0000 150.0040 150.0240 150.0480 150.0720 150.1040
	totalWidths			= 50.0000 50.0090 50.0100 50.0150 50.0210 50.0250 50.0330 50.0370 50.0450 50.0490 50.0980 50.1290 50.1310 100.0000 150.0000 150.0040 150.0240 150.0480 150.0720 150.1040
	sd_crest			= -0.2912
	sd_flowArea			= 1.0473561480652571
	sd_totalArea		= 1.0474
	sd_baseLevel		= -0.7912269469444795
	main				= 50.1040
	floodPlain1			= 100.0000
	floodPlain2			= 0.0000
	groundlayerUsed		= 0
	groundLayer			= 0.000
    ```

=== "dhydro"

    ```ini
    [Definition]
	Id					= channel1_125.000
	Type				= zwRiver
	Thalweg				= 0.000
	numLevels			= 20
	levels				= -2.0249 -1.9486 -1.9476 -1.8829 -1.8261 -1.7498 -1.6281 -1.5567 -1.4315 -1.3622 -0.5710 -0.0747 -0.0391 -0.0216 -0.0049 0.0695 0.4215 0.8350 1.2430 1.7807
	flowWidths			= 50.0000 50.0090 50.0100 50.0150 50.0210 50.0250 50.0330 50.0370 50.0450 50.0490 50.0980 50.1290 50.1310 50.1320 150.0000 150.0040 150.0240 150.0480 150.0720 150.1040
	totalWidths			= 50.0000 50.0090 50.0100 50.0150 50.0210 50.0250 50.0330 50.0370 50.0450 50.0490 50.0980 50.1290 50.1310 100.0000 150.0000 150.0040 150.0240 150.0480 150.0720 150.1040
	leveeCrestLevel		= -0.2912
	leveeFlowArea		= 1.0473561480652571
	leveeTotalArea		= 1.0474
	levelBaseLevel		= -0.7912269469444795
	mainWidth			= 50.1040
	fp1Width			= 100.0000
	fp2Width			= 0.0000
    ```



### roughness-x.ini 

This file contains roughness coefficients for a section of the cross-section. Valid sections are "main", "floodplain1" and "floodplain2". FM2PROF always output the values as Chézy coefficient. 


**Example:**
=== "dflow1d"
    ```ini
    [Content]
        sectionId             = Main
        flowDirection         = False
        interpolate           = 1
        globalType            = 1
        globalValue           = 45

    [BranchProperties]
        branchId              = Channel1
        roughnessType         = 1
        functionType          = 2
        numLevels             = 42
        levels                = -2.4028 -2.3028 -2.2028 -2.1028 -2.0028 -1.9028 -1.8028 -1.7028 -1.6028 -1.5028 -1.4028 -1.3028 -1.2028 -1.1028 -1.0028 -0.9028 -0.8028 -0.7028 -0.6028 -0.5028 -0.4028 -0.3028 -0.2028 -0.1028 -0.0028 0.0972 0.1972 0.2972 0.3972 0.4972 0.5972 0.6972 0.7972 0.8972 0.9972 1.0972 1.1972 1.2972 1.3972 1.4972 1.5972 1.6972

    [Definition]
        branchId              = Channel1
        chainage              = 125.0
        values                = 22.71 22.71 22.71 22.71 22.71 23.47 25.93 27.59 28.86 29.91 30.79 31.57 32.26 32.88 33.45 33.98 34.47 34.92 35.35 35.75 36.13 36.49 36.84 37.17 37.48 37.79 38.08 38.36 38.62 38.89 39.14 39.38 39.62 39.85 40.07 40.29 40.5 40.71 40.91 41.11 41.3 41.49
    ```

=== "dhydro"

    ```ini
    [Global]
        frictionId             = Main
        frictionType            = Chezy
        frictionValue           = 45

    [Branch]
        branchId              = Channel1
        frictionType         = Chezy
        functionType          = Waterlevel
        numLevels             = 42
        levels                = -2.4028 -2.3028 -2.2028 -2.1028 -2.0028 -1.9028 -1.8028 -1.7028 -1.6028 -1.5028 -1.4028 -1.3028 -1.2028 -1.1028 -1.0028 -0.9028 -0.8028 -0.7028 -0.6028 -0.5028 -0.4028 -0.3028 -0.2028 -0.1028 -0.0028 0.0972 0.1972 0.2972 0.3972 0.4972 0.5972 0.6972 0.7972 0.8972 0.9972 1.0972 1.1972 1.2972 1.3972 1.4972 1.5972 1.6972
        numLocations          = 14
        chainage               = 125.0000 225.0000 325.0000 425.0000 525.0000 625.0000 725.0000 825.0000 925.0000 1025.0000 1125.0000 1225.0000 1325.0000 1425.0000
        frictionValues                = 22.7097 22.7097 22.7097 22.7097 22.7097 23.4671 25.9284 27.5925 28.8645 29.9081 30.7947 31.5704 32.2605 32.8842 33.4534 33.9782 34.4652 34.9203 35.3474 35.7501 36.1314 36.4936 36.8386 37.1682 37.4836 37.7854 38.0757 38.3553 38.6250 38.8857 39.1379 39.3824 39.6195 39.8498 40.0737 40.2915 40.5037 40.7105 40.9122 41.1091 41.3015 41.4894
    22.7097 22.7097 22.7097 22.7097 22.7097 24.4354 26.5433 28.0504 29.2338 30.2186 31.0641 31.8090 32.4751 33.0796 33.6330 34.1445 34.6203 35.0657 35.4842 35.8795 36.2541 36.6105 36.9502 37.2749 37.5855 37.8843 38.1715 38.4485 38.7156 38.9739 39.2239 39.4661 39.7011 39.9294 40.1513 40.3673 40.5776 40.7827 40.9826 41.1779 41.3687 41.5551
    22.7097 22.7097 22.7097 22.7097 22.7097 25.2428 27.0931 28.4730 29.5806 30.5140 31.3224 32.0391 32.6829 33.2692 33.8078 34.3068 34.7720 35.2079 35.6184 36.0066 36.3749 36.7254 37.0601 37.3800 37.6866 37.9814 38.2654 38.5390 38.8033 39.0589 39.3063 39.5463 39.7791 40.0053 40.2253 40.4394 40.6480 40.8514 41.0498 41.2435 41.4328 41.6179
    22.7097 22.7097 22.7097 22.7097 23.4664 25.9278 27.5927 28.8645 29.9081 30.7947 31.5704 32.2606 32.8842 33.4535 33.9783 34.4653 34.9204 35.3475 35.7502 36.1315 36.4937 36.8387 37.1683 37.4838 37.7864 38.0774 38.3578 38.6283 38.8896 39.1424 39.3874 39.6249 39.8555 40.0797 40.2977 40.5100 40.7168 40.9186 41.1154 41.3077 41.4956 41.6171
    22.7097 22.7097 22.7097 22.7097 24.4352 26.5437 28.0502 29.2337 30.2186 31.0641 31.8091 32.4752 33.0796 33.6330 34.1446 34.6204 35.0657 35.4843 35.8796 36.2542 36.6106 36.9502 37.2751 37.5858 37.8847 38.1721 38.4491 38.7164 38.9749 39.2250 39.4675 39.7026 39.9311 40.1531 40.3693 40.5797 40.7849 40.9849 41.1803 41.3711 41.5576 41.6163
    22.7097 22.7097 22.7097 22.7097 25.2430 27.0933 28.4729 29.5806 30.5140 31.3225 32.0392 32.6829 33.2692 33.8079 34.3068 34.7720 35.2080 35.6185 36.0067 36.3750 36.7255 37.0602 37.3802 37.6868 37.9816 38.2656 38.5392 38.8036 39.0592 39.3067 39.5467 39.7795 40.0058 40.2259 40.4401 40.6487 40.8522 41.0507 41.2445 41.4338 41.6164 41.6164
    22.7097 22.7097 22.7097 23.4659 25.9273 27.5928 28.8644 29.9081 30.7948 31.5705 32.2606 32.8843 33.4535 33.9784 34.4654 34.9205 35.3475 35.7503 36.1316 36.4938 36.8388 37.1685 37.4840 37.7865 38.0775 38.3579 38.6283 38.8897 39.1425 39.3875 39.6250 39.8557 40.0799 40.2979 40.5102 40.7172 40.9189 41.1158 41.3081 41.4961 41.6168 41.6168
    22.7097 22.7097 22.7097 24.4351 26.5439 28.0501 29.2337 30.2186 31.0641 31.8091 32.4752 33.0797 33.6331 34.1447 34.6205 35.0658 35.4844 35.8797 36.2544 36.6107 36.9504 37.2752 37.5859 37.8848 38.1721 38.4492 38.7165 38.9750 39.2251 39.4675 39.7027 39.9312 40.1532 40.3694 40.5798 40.7850 40.9851 41.1805 41.3713 41.5578 41.6174 41.6174
    22.7097 22.7097 22.7170 25.2431 27.0935 28.4729 29.5805 30.5141 31.3225 32.0392 32.6830 33.2693 33.8080 34.3069 34.7721 35.2081 35.6187 36.0068 36.3751 36.7257 37.0603 37.3803 37.6869 37.9818 38.2656 38.5393 38.8036 39.0592 39.3067 39.5467 39.7796 40.0059 40.2259 40.4401 40.6488 40.8523 41.0507 41.2446 41.4339 41.6181 41.6181 41.6181
    22.7097 22.7097 23.4656 25.9269 27.5930 28.8644 29.9081 30.7948 31.5705 32.2607 32.8843 33.4536 33.9784 34.4655 34.9206 35.3476 35.7505 36.1317 36.4939 36.8389 37.1686 37.4841 37.7866 38.0776 38.3580 38.6284 38.8898 39.1426 39.3876 39.6251 39.8558 40.0799 40.2980 40.5103 40.7172 40.9190 41.1159 41.3082 41.4961 41.6188 41.6188 41.6188
    22.7097 22.7097 24.4350 26.5434 28.0500 29.2336 30.2186 31.0642 31.8092 32.4753 33.0797 33.6332 34.1448 34.6206 35.0659 35.4845 35.8799 36.2545 36.6108 36.9505 37.2754 37.5860 37.8849 38.1722 38.4493 38.7165 38.9750 39.2251 39.4676 39.7027 39.9312 40.1533 40.3694 40.5798 40.7850 40.9851 41.1805 41.3713 41.5579 41.6196 41.6196 41.6196
    22.7097 22.7217 25.2429 27.0938 28.4729 29.5805 30.5141 31.3225 32.0392 32.6830 33.2694 33.8081 34.3071 34.7722 35.2082 35.6188 36.0070 36.3753 36.7258 37.0604 37.3804 37.6871 37.9819 38.2657 38.5394 38.8037 39.0593 39.3068 39.5468 39.7796 40.0059 40.2260 40.4402 40.6488 40.8523 41.0508 41.2446 41.4340 41.6191 41.6205 41.6205 41.6205
    22.7097 23.4654 25.9266 27.5928 28.8644 29.9081 30.7948 31.5705 32.2608 32.8844 33.4537 33.9786 34.4656 34.9207 35.3478 35.7506 36.1319 36.4941 36.8391 37.1687 37.4842 37.7867 38.0777 38.3581 38.6285 38.8899 39.1427 39.3876 39.6251 39.8558 40.0800 40.2980 40.5103 40.7172 40.9190 41.1159 41.3082 41.4962 41.6213 41.6213 41.6213 41.6213
    22.7097 24.4351 26.5430 28.0499 29.2336 30.2186 31.0642 31.8092 32.4754 33.0798 33.6333 34.1449 34.6207 35.0661 35.4846 35.8801 36.2547 36.6110 36.9507 37.2755 37.5862 37.8851 38.1723 38.4493 38.7166 38.9751 39.2252 39.4676 39.7028 39.9312 40.1533 40.3694 40.5799 40.7850 40.9852 41.1805 41.3713 41.5579 41.6222 41.6222 41.6222 41.6222
    ```


### volumes.csv

This file contains a comparison between 2D model volumes and the reconstructed 1D cross-section volumes.

**Columns:**

| Column Name | Description |
|-------------|-------------|
| `id` | Cross-section identifier |
| `z` | Water level (elevation) |
| `2D_total_volume` | Total storage volume from 2D model |
| `2D_flow_volume` | Flow (conveyance) volume from 2D model |
| `2D_wet_area` | Wetted area from 2D model |
| `2D_flow_area` | Flow area from 2D model |
| `1D_total_volume_sd` | Total volume for 1D cross-section including levee (summer dike)|
| `1D_total_volume` | Total storage volume for 1D cross-section |
| `1D_flow_volume_sd` | Flow volume for 1D cross-section including levee (summer dike)|
| `1D_flow_volume` | Flow volume for 1D cross-section |
| `1D_total_width` | Total width of 1D cross-section |
| `1D_flow_width` | Flow width of 1D cross-section |

**Example:**
```csv
id,z,2D_total_volume,2D_flow_volume,2D_wet_area,2D_flow_area,1D_total_volume_sd,1D_total_volume,1D_flow_volume_sd,1D_flow_volume,1D_total_width,1D_flow_width
CS_001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
CS_001,0.5,125.5,100.2,250.0,200.0,124.8,125.0,99.8,100.0,250.0,200.0
CS_001,1.0,502.3,400.5,502.0,401.0,501.5,502.0,399.9,400.0,502.0,401.0
...
```
