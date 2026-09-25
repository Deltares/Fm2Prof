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
        "total_width": [
            0.5000, 1.1586, 4.5337, 7.7060, 13.7614, 20.6931, 23.5097, 24.1656, 24.3621, 25.2960,
            25.3120, 27.9369, 28.2711, 29.4732, 30.9638, 31.1367, 31.8644, 32.9435, 33.6202, 33.7372,
        ],
        "levels": [
            24.0133, 24.1267, 24.2400, 24.3166, 24.4169, 24.6033, 24.8677, 25.1774, 25.9285, 26.4067,
            27.2827, 27.7572, 28.0198, 28.1867, 28.5005, 28.7000, 28.8582, 28.8693, 28.9987, 29.3749,
        ],
    },
    "expected_dflow1d_files": [
        "CrossSectionLocations.ini",
        "CrossSectionDefinitions.ini",
    ],
    "expected_dhydro_files": [
        "crsloc.ini",
        "crsdef.ini",
    ]},
    {
    "name": "case_20_geotif",
    "inifile": "cases/case_20_only_elevation/fm2prof_config_geotif.ini",
    "expected_cross_section": {
        "total_width": [
            0.5000, 0.6300, 1.2211, 4.9085, 10.8084, 15.5678, 18.6306, 21.3589, 22.4792, 23.8051,
            24.3371, 25.1310, 25.5622, 25.9773, 27.2024, 28.1091, 29.8017, 30.6400, 32.8887, 33.3400,
        ],
        "levels": [
            23.9658, 24.0718, 24.1150, 24.2359, 24.3717, 24.4658, 24.5499, 24.6808, 24.7781, 24.9814,
            25.2050, 26.7276, 27.2121, 27.5285, 27.7317, 27.9379, 28.3804, 28.6309, 29.0069, 29.4173,
        ],
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
cases_idealised = [c for c in cases if "case_02" in c["name"]]
cases_elevation_only = [c for c in cases if "case_20" in c["name"]]

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

    @pytest.mark.parametrize("case", cases_elevation_only)
    def test_elevation_only_css_matches_expectations(self, case):
        """Test that the elevation-only case produces a cross-section matching the expected vectors exactly."""
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
        #css["levels"] = [lvl - min(css["levels"]) for lvl in css["levels"]]

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

