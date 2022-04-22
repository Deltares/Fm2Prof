Numerical methods
======================


.. _simplify_css:

Cross-section simplification algorithm
............................................

.. automethod:: fm2prof.CrossSection.CrossSection.reduce_points


.. _identify_lakes:

Lake identification algorithm
...................................

.. automethod:: fm2prof.CrossSection.CrossSection._identify_lakes


.. _distinguish_storage:

Flow & storage seperation algorithm
...................................

.. automethod:: fm2prof.CrossSection.CrossSection._distinguish_flow_from_storage


.. _wl_dependent_css:

Water level dependent geometry algorithm
.........................................

.. automethod:: fm2prof.CrossSection.CrossSection._compute_css_above_z0



.. _wl_independent_css:

Water level independent geometry algorithm
............................................

.. automethod:: fm2prof.CrossSection.CrossSection._extend_css_below_z0


.. _sd_optimisation:

Summer dike optimisation algorithm
............................................

.. automethod:: fm2prof.CrossSection.CrossSection.calculate_correction


.. _section_classification_variance:

Variance based classification
...................................

.. automethod:: fm2prof.Fm2ProfRunner.Fm2ProfRunner._classify_roughness_sections_by_variance


