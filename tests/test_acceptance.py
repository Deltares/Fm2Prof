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

# Expected D-Flow 1D output files
EXPECTED_DFLOW1D_FILES = [
    "CrossSectionLocations.ini",
    "CrossSectionDefinitions.ini",
    "roughness-Main.ini",
    "roughness-FloodPlain1.ini",
    "volumes.csv",
]

# Expected D-Hydro output files
EXPECTED_DHYDRO_FILES = [
    "crsloc.ini",
    "crsdef.ini",
    "roughness-Main.ini",
    "roughness-FloodPlain1.ini",
    "volumes.csv",
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
        css_def_file = project.get_output_directory() / "dflow1d" / "CrossSectionDefinitions.ini"
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

    @pytest.mark.parametrize("case", cases)
    def test_all_dflow1d_output_files_created(self, case):
        """Test that all expected D-Flow 1D output files are created after running a project."""
        # 1. Set up test data
        inifile = TestUtils.get_local_test_file(case.get("inifile"))

        # 2. Run the project
        project = Project(inifile)
        success = project.run(overwrite=True)

        # 3. Verify the project ran successfully
        assert success, f"Project run failed for case: {case.get('name')}"

        # 4. Get the output directory for D-Flow 1D format
        output_dir = project.get_output_directory() / "dflow1d"

        # 5. Check that the output directory exists
        assert output_dir.exists(), f"Output directory does not exist: {output_dir}"

        # 6. Check that all expected files are created
        missing_files = []
        for expected_file in EXPECTED_DFLOW1D_FILES:
            file_path = output_dir / expected_file
            if not file_path.exists():
                missing_files.append(expected_file)

        # 7. Assert that no files are missing
        assert not missing_files, (
            f"Missing expected output files for case '{case.get('name')}': {missing_files}"
        )

    @pytest.mark.parametrize("case", cases)
    def test_all_dhydro_output_files_created(self, case):
        """Test that all expected D-Hydro output files are created after running a project."""
        # 1. Set up test data
        inifile = TestUtils.get_local_test_file(case.get("inifile"))

        # 2. Run the project
        project = Project(inifile)
        success = project.run(overwrite=True)

        # 3. Verify the project ran successfully
        assert success, f"Project run failed for case: {case.get('name')}"

        # 4. Get the output directory for D-Flow 1D format
        output_dir = project.get_output_directory() / "dhydro"

        # 5. Check that the output directory exists
        assert output_dir.exists(), f"Output directory does not exist: {output_dir}"

        # 6. Check that all expected files are created
        missing_files = []
        for expected_file in EXPECTED_DHYDRO_FILES:
            file_path = output_dir / expected_file
            if not file_path.exists():
                missing_files.append(expected_file)

        # 7. Assert that no files are missing
        assert not missing_files, (
            f"Missing expected output files for case '{case.get('name')}': {missing_files}"
        )
