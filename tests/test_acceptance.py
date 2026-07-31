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
    {
    "name": "case_20_elevation_only",
    "inifile": "cases/case_20_only_elevation/fm2prof_config.ini",
    "expected_cross_section": {
        "total_width": [0.50, 1.21, 4.53, 15.59, 21.87, 23.51, 24.17, 24.37, 25.09, 25.32, 25.33, 26.02, 26.68, 27.29, 29.10, 30.96, 31.86, 32.94, 33.62, 33.74],
        "levels":      [0.00, 0.12, 0.23,  0.44,  0.70,  0.85,  1.16,  1.92,  2.21,  2.48,  3.27,  3.73,  3.76,  4.00,  4.14,  4.49,  4.85,  4.86,  4.99,  5.36],
    },
    "expected_dflow1d_files": [
        "CrossSectionLocations.ini",
        "CrossSectionDefinitions.ini",
    ],
    "expected_dhydro_files": [
        "crsloc.ini",
        "crsdef.ini",
    ]},
]

# Subset used for idealised cross-section comparison (excludes elevation-only case)
cases_idealised = [c for c in cases if c["name"] != "case_20_elevation_only"]

# Expected D-Flow 1D output files
EXPECTED_DFLOW1D_FILES = [
    "CrossSectionLocations.ini",
    "CrossSectionDefinitions.ini",
    "roughness-Main.ini",
    "volumes.csv",
]

# Expected D-Hydro output files
EXPECTED_DHYDRO_FILES = [
    "crsloc.ini",
    "crsdef.ini",
    "roughness-Main.ini",
    "volumes.csv",
]

class TestAcceptance:

    @pytest.fixture(autouse=True)
    def clear_polygon_caches(self):
        """Delete any cached region/section classification files before each test.

        Cache files are placed next to the map file and have the pattern
        ``<mapfile>.region_cache.json`` and ``<mapfile>.section_cache.json``.
        Removing them ensures each test run performs a fresh classification.
        """
        cache_dir = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput")
        for cache_file in cache_dir.glob("*_cache.json"):
            cache_file.unlink(missing_ok=True)

    @pytest.mark.parametrize("case", cases_idealised)
    def test_generated_css_match_expected(self, case):
        # 1. Set up test data and expectations
        tolerated_max_level_error = 0.05 # meters
        inifile = TestUtils.get_local_test_file(case.get("inifile"))

        # 2. run case
        project = Project(inifile)
        project.set_output_directory(project.get_output_directory() / case.get("name"))
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

    def test_elevation_only_css_matches_expectations(self):
        """Test that the elevation-only case produces a cross-section matching the expected vectors exactly."""
        case = next(c for c in cases if c["name"] == "case_20_elevation_only")
        inifile = TestUtils.get_local_test_file(case.get("inifile"))

        # Run case
        project = Project(inifile)
        project.set_output_directory(project.get_output_directory() / case.get("name"))
        success = project.run(overwrite=True)
        assert success

        # Parse output
        css_def_file = project.get_output_directory() / "dflow1d" / "CrossSectionDefinitions.ini"
        css_def = VisualiseOutput.parse_cross_section_definition_file(css_def_file)
        css = css_def[0]

        # Normalise levels to start at 0
        css["levels"] = [lvl - min(css["levels"]) for lvl in css["levels"]]

        expected = case.get("expected_cross_section")
        assert css["total_width"] == pytest.approx(expected["total_width"], abs=0.01), \
            f"total_width mismatch:\n  actual  : {css['total_width']}\n  expected: {expected['total_width']}"
        assert css["levels"] == pytest.approx(expected["levels"], abs=0.01), \
            f"levels mismatch:\n  actual  : {css['levels']}\n  expected: {expected['levels']}"

    @pytest.mark.parametrize("case", cases)
    def test_all_dflow1d_output_files_created(self, case):
        """Test that all expected D-Flow 1D output files are created after running a project."""
        # 1. Set up test data
        inifile = TestUtils.get_local_test_file(case.get("inifile"))

        # 2. Run the project
        project = Project(inifile)
        project.set_output_directory(project.get_output_directory() / case.get("name"))
        success = project.run(overwrite=True)

        # 3. Verify the project ran successfully
        assert success, f"Project run failed for case: {case.get('name')}"

        # 4. Get the output directory for D-Flow 1D format
        output_dir = project.get_output_directory() / "dflow1d"

        # 5. Check that the output directory exists
        assert output_dir.exists(), f"Output directory does not exist: {output_dir}"

        # 6. Check that all expected files are created
        expected_files = case.get("expected_dflow1d_files", EXPECTED_DFLOW1D_FILES)
        missing_files = [f for f in expected_files if not (output_dir / f).exists()]

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
        expected_files = case.get("expected_dhydro_files", EXPECTED_DHYDRO_FILES)
        missing_files = [f for f in expected_files if not (output_dir / f).exists()]

        # 7. Assert that no files are missing
        assert not missing_files, (
            f"Missing expected output files for case '{case.get('name')}': {missing_files}"
        )

