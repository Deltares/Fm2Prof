Settings
==========

Below is a list of all parameters, categorized by which function they affect. All parameters are listed in the configuration file in the `[parameters]` section. 

.. role:: sep (strong)
.. role:: aspect (emphasis)



.. _parameter_casename:
.. container:: dl-parameters

    CaseName
        :aspect:`Type:` |type_casename| 
        :sep:`|` :aspect:`Default:` |default_casename|
        :sep:`|` :aspect:`Hint:` |hint_casename|
                      
.. _parameter_maximumpointsinprofile:
.. container:: dl-parameters

    MaximumPointsInProfile
        :aspect:`Type:` |type_maximumpointsinprofile| 
        :sep:`|` :aspect:`Default:` |default_maximumpointsinprofile|
        :sep:`|` :aspect:`Hint:` |hint_maximumpointsinprofile|
        
        This parameter controls how many point described cross-section geometry. 

        See: :ref:`simplify_css`

.. _parameter_absolutevelocitythreshold:
.. container:: dl-parameters

    AbsoluteVelocityThreshold
        :aspect:`Type:`                |type_absolutevelocitythreshold| 
        :sep:`|` :aspect:`Default:` |default_absolutevelocitythreshold|
        :sep:`|` :aspect:`Hint:`    |hint_absolutevelocitythreshold|

        These thresholds are used to distinguish conveyance (effective flow) from storage (dead zones) areas. Parts of the cross-section that do not contribute to flow are called 'storage area'. A cell is considered part of the flow area if all of the following conditions are met:

        - the water depth is higher than 0
        - the (depth-averaged) velocity is larger than `AbsoluteVelocityThreshold`
        - the (depth-averaged) velocity is larger than `RelativeVelocityThreshold` multiplied by the average velocity in the cross-section

        These conditions are checked for each timestep in the 2D model output.

        Used in: :ref:`distinguish_storage`

.. _parameter_relativevelocitythreshold:
.. container:: dl-parameters

    RelativeVelocityThreshold
        :aspect:`Type:`                |type_relativevelocitythreshold| 
        :sep:`|` :aspect:`Default:` |default_relativevelocitythreshold|
        :sep:`|` :aspect:`Hint:`    |hint_relativevelocitythreshold|
    
        See :ref:`AbsoluteVelocityThreshold <parameter_absolutevelocitythreshold>`

        Used in: :ref:`distinguish_storage`

.. _parameter_minimumdepththreshold:
.. container:: dl-parameters

    MinimumDepthThreshold
        :aspect:`Type:`                |type_minimumdepththreshold| 
        :sep:`|` :aspect:`Default:` |default_minimumdepththreshold|
        :sep:`|` :aspect:`Hint:`    |hint_minimumdepththreshold|

.. _parameter_bedlevelcriterium:
.. container:: dl-parameters

    BedlevelCriterium
        :aspect:`Type:`                |type_bedlevelcriterium| 
        :sep:`|` :aspect:`Default:` |default_bedlevelcriterium|
        :sep:`|` :aspect:`Hint:`    |hint_bedlevelcriterium|

.. _parameter_laketimesteps:
.. container:: dl-parameters

    LakeTimesteps
        :aspect:`Type:`                |type_laketimesteps| 
        :sep:`|` :aspect:`Default:` |default_laketimesteps|
        :sep:`|` :aspect:`Hint:`    |hint_laketimesteps|

        This parameter defines the amount of 2D model timesteps used to determine if a wet cell is hydraulically connected to the main channel. It recommended to only deviate from the default value if lakes are misidentified, and to generally keep the number low. A good test to set the value for this parameter is to visualise the 'IsLake' attribute (see :ref:`diagnosis`). Too high values lead to overestimation of the total volume: everything that is identified as 'main channel' will be used in the so-called 'water-level independent' part of the geometry identification. 

        Used in: :ref:`identify_lakes`

.. _parameter_extrapolatestorage:
.. container:: dl-parameters

    ExtrapolateStorage
        :aspect:`Type:`                |type_extrapolatestorage| 
        :sep:`|` :aspect:`Default:` |default_extrapolatestorage|
        :sep:`|` :aspect:`Hint:`    |hint_extrapolatestorage|

        Used in :ref:`wl_independent_css`


.. _parameter_sdcorrection:
.. container:: dl-parameters

    SDCorrection
        :aspect:`Type:`                |type_sdcorrection| 
        :sep:`|` :aspect:`Default:` |default_sdcorrection|
        :sep:`|` :aspect:`Hint:`    |hint_sdcorrection|

    Used in :ref:`sd_optimisation`
    
.. _parameter_sdfloodplainbase:
.. container:: dl-parameters

    SDFloodplainBase
        :aspect:`Type:`                |type_sdfloodplainbase| 
        :sep:`|` :aspect:`Default:` |default_sdfloodplainbase|
        :sep:`|` :aspect:`Hint:`    |hint_sdfloodplainbase|

    Used in :ref:`sd_optimisation`

.. _parameter_sdtransitionheight:
.. container:: dl-parameters

    SDTransitionHeight
        :aspect:`Type:`                |type_sdtransitionheight| 
        :sep:`|` :aspect:`Default:` |default_sdtransitionheight|
        :sep:`|` :aspect:`Hint:`    |hint_sdtransitionheight|

    Used in :ref:`sd_optimisation`

.. _parameter_sdoptimisationmethod:
.. container:: dl-parameters

    SDOptimisationMethod
        :aspect:`Type:`                |type_sdoptimisationmethod| 
        :sep:`|` :aspect:`Default:` |default_sdoptimisationmethod|
        :sep:`|` :aspect:`Hint:`    |hint_sdoptimisationmethod|

    Used in :ref:`sd_optimisation`

.. _parameter_frictionweighingmethod:
.. container:: dl-parameters

    FrictionWeighingMethod
        :aspect:`Type:`                |type_frictionweighingmethod| 
        :sep:`|` :aspect:`Default:` |default_frictionweighingmethod|
        :sep:`|` :aspect:`Hint:`    |hint_frictionweighingmethod|

.. _parameter_exportmapfiles:
.. container:: dl-parameters

    ExportMapFiles
        :aspect:`Type:`                |type_exportmapfiles| 
        :sep:`|` :aspect:`Default:` |default_exportmapfiles|
        :sep:`|` :aspect:`Hint:`    |hint_exportmapfiles|
        
        If this parameter is set to `True`, |project| will output two additional geojson files. These additional files contain diagnostic information for each 2D model output (e.g. to which cross-section a 2D point is assigned). However, for large models this output can be quite large. For detailed diagnosis, combine this parameter with :ref:`CssSelection <parameter_cssselection>`


.. _parameter_cssselection:
.. container:: dl-parameters

    CssSelection
        :aspect:`Type:`                |type_cssselection| 
        :sep:`|` :aspect:`Default:` |default_cssselection|
        :sep:`|` :aspect:`Hint:`    |hint_cssselection|

        This parameter is used to run |project| for a subset of cross-sections in the `CrossSectionLocationFile`. Its main use is for diagnostic purposes. For example, if you want to closely inspect the 54th, 76th and 89th cross-section, use:

    .. code-block:: text
    
        CssSelection = [54, 76, 89]
        ExportMapFiles = True


.. _parameter_skipmaps:
.. container:: dl-parameters

    SkipMaps
        :aspect:`Type:`                |type_skipmaps| 
        :sep:`|` :aspect:`Default:` |default_skipmaps|
        :sep:`|` :aspect:`Hint:`    |hint_skipmaps|

        This parameter is used to skip the first number of output timesteps ('maps') in the 2D model output. This parameter can be useful if the 2D model is not completely in equilibrium at the start of the computation (e.g. falling water levels in the first few timesteps). However, it tests have shown that it is far better to carefully initialize the 2D model, than to skip the first few steps with this parameter. 

.. _parameter_classificationmethod:
.. container:: dl-parameters

    ClassificationMethod
        :aspect:`Type:` |type_classificationmethod| 
        :sep:`|` :aspect:`Default:` |default_classificationmethod|
        :sep:`|` :aspect:`Hint:`    |hint_classificationmethod|

        This parameter defines the method used to classify the region and section of each output point in the 2D output. |project| has built-in classification method. However these methods are not efficient for large models. The currently supported 'DeltaShell' method consists of a manual work-around. 

        .. note:: 
            This approach is not well documented as it involves some manual work. We expect to automise this and make it more user friendly in a future update. 


.. _parameter_minimumtotalwidth:
.. container:: dl-parameters

    MinimumTotalWidth
        :aspect:`Type:`                |type_minimumtotalwidth| 
        :sep:`|` :aspect:`Default:` |default_minimumtotalwidth|
        :sep:`|` :aspect:`Hint:`    |hint_minimumtotalwidth|

        The purpose of this parameter is to prevent instabilities in 1D Solvers if they are presented with a (near) zero width. It is recommended to keep this value small (at default). 


