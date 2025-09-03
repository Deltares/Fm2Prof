import numpy as np
import pytest

from fm2prof.fm2prof_runner import Project
from fm2prof.utils import VisualiseOutput
from tests.TestUtils import TestUtils

_root_output_dir = None

# Test data to be used
cases = [{
    "name": "case_02_compound",
    "inifile": "cases/case_02_compound/fm2prof_config.ini",
    "expected_cross_section": {
        "total_width": [20, 80],
        "levels": [0, 2],
    }},
    {
    "name": "case_02_compound_with_region",
    "inifile": "cases/case_02_compound/fm2prof_config_with_region.ini",
    "expected_cross_section": {
        "total_width": [20, 80],
        "levels": [0, 2],
    }},
    {
    "name": "case_02_compound_with_region_and_section",
    "inifile": "cases/case_02_compound/fm2prof_config_with_region_and_section.ini",
    "expected_cross_section": {
        "total_width": [20, 80],
        "levels": [0, 2],
    }},
]


class TestAcceptance:

    @pytest.mark.parametrize("case", cases)
    def test_generated_css_match_expected(self, case):
        # 1. Set up test data and expectations
        tolerated_max_level_error = 0.05 # meters
        inifile = TestUtils.get_local_test_file(case.get("inifile"))

        # 2. run case
        project = Project(inifile)

        success = project.run(overwrite=True)

        # 3. verify output
        assert success

        # 4. get output
        css_def_file = project.get_output_directory() / "CrossSectionDefinitions.ini"
        css_def = VisualiseOutput.parse_cross_section_definition_file(css_def_file)

        # 5. verify output
        # normalize levels
        css = css_def[0] # test only the first cross-section
        css["levels"] = [lvl - min(css["levels"]) for lvl in css["levels"]]

        css_test_points = case.get("expected_cross_section")
        # get the error in width for the given expected points
        max_lvl_error = 0
        for lvl, width in zip(css_test_points["levels"], css_test_points["total_width"], strict=True):
            expected_lvl = np.interp(width,
                                    css.get("total_width"),
                                    css.get("levels"))

            max_lvl_error = max(max_lvl_error, abs(expected_lvl - lvl))

        assert max_lvl_error < tolerated_max_level_error
