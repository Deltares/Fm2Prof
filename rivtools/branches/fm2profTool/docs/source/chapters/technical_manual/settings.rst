Settings
==========

This chapter details the input files and parameters settings. 

Input files
---------------------------------


Parameters
---------------------------------

Below is a list of all parameters, categorized by which function they affect. All parameters are listed in the configuration file in the `[parameters]` section. 

General 
.........

- CaseName                  
- ClassificationMethod      

Geometry
.........

- MaximumPointsInProfile    
- MinimumDepthThreshold     
- BedlevelCriterium         
- MinimumTotalWidth         
- LakeTimesteps             

Storage
.........

- AbsoluteVelocityThreshold 
- RelativeVelocityThreshold 
- ExtrapolateStorage        

Summer dike / volume correction
................................
- SDCorrection              
- SDFloodplainBase          
- SDTransitionHeight        
- SDOptimisationMethod      

Roughness
...........
- FrictionWeighingMethod    

Output
...........

- ExportMapFiles            
- CssSelection              
- SkipMaps                  



Recommended settings
^^^^^^^^^^^^^^^^^^^^^^^^^^

The recommendations detailed in this section are based on the Meuse River pilot project. 

LakeTimesteps
    This parameter defines the amount of 2D model timesteps used to determine if a wet cell is hydraulically connected to the main channel. It recommended to only deviate from the default value if lakes are misidentified, and to generally keep the number low. A good test to set the value for this parameter is to visualise the 'IsLake' attribute (see :ref:`diagnosis`). Too high values lead to overestimation of the total volume: everything that is identified as 'main channel' will be used in the so-called 'water-level independent' part of the geometry identification. 

MinimumTotalWidth
    The purpose of this parameter is to prevent instabilities in 1D Solvers if they are presented with a (near) zero width. It is recommended to keep this value small (at default). 

ClassificationMethod
    This parameter defines the method used to classify the region and section of each output point in the 2D output. |project| has built-in classification method. However these methods are not efficient for large models. The currently supported 'DeltaShell' method consists of a manual work-around. 

    .. note:: 
        This approach is not well documented as it involves some manual work. We expect to automise this and make it more user friendly in a future update. 

AbsoluteVelocityThreshold / RelativeVelocityThreshold 
    These thresholds are used to distinguish conveyance (effective flow) from storage (dead zones) areas. Parts of the cross-section that do not contribute to flow are called 'storage area'. A cell is considered part of the flow area if all of the following conditions are met:

        - the water depth is higher than 0
        - the (depth-averaged) velocity is larger than `AbsoluteVelocityThreshold`
        - the (depth-averaged) velocity is larger than `RelativeVelocityThreshold` multiplied by the average velocity in the cross-section

    These conditions are checked for each timestep in the 2D model output.

SkipMaps
    This parameter is used to skip the first number of output timesteps ('maps') in the 2D model output. This parameter can be useful if the 2D model is not completely in equilibrium at the start of the computation (e.g. falling water levels in the first few timesteps). However, it tests have shown that it is far better to carefully initialize the 2D model, than to skip the first few steps with this parameter. 

ExportMapFiles
    If this parameter is set to `True`, |project| will output two additional geojson files. These additional files contain diagnostic information for each 2D model output (e.g. to which cross-section a 2D point is assigned). However, for large models this output can be quite large. For detailed diagnosis, combine this parameter with `CssSelection`

CssSelection
    This parameter is used to run |project| for a subset of cross-sections in the `CrossSectionLocationFile`. Its main use is for diagnostic purposes. For example, if you want to closely inspect the 54th, 76th and 89th cross-section, use:

    .. code-block:: text
    
        CssSelection = [54, 76, 89]
        ExportMapFiles = True

