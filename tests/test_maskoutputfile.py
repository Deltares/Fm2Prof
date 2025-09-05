from pathlib import Path

import geojson
import pytest

from fm2prof.mask_output_file import (
    create_mask_point,
    write_mask_output_file,
    read_mask_output_file,
    validate_extension,
)
from tests.TestUtils import TestUtils


class Test_create_mask_point:
    def test_when_no_coords_does_raise(self):
        # 1. Set up test model
        coords = None
        properties = None

        # 2. Set up initial expectations
        expected_error = "coords cannot be empty."

        # 3. Do test
        with pytest.raises(ValueError, match=expected_error):
            create_mask_point(coords, properties)

    @pytest.mark.parametrize("coords_values", [(4.2, 2.4), (4.2, 2.4, 42)])
    def test_when_valid_coords_does_not_raise(self, coords_values: tuple):
        # 1. Set up test model
        coords = coords_values
        properties = None

        # 2. Do test
        mask_point = create_mask_point(coords, properties)

        # 3. Verify final expectations
        assert mask_point is not None, "No mask_point generated"


class Test_validate_extension:
    def test_when_none_file_path_does_raise(self):
        with pytest.raises(TypeError, match=f"file_path should be string or Path, not {type(None)}"):
            validate_extension(None)

    def test_when_invalid_extension_raises_expected_exception(self):
        # 1. Set up test data
        file_path = Path("test.sjon")

        # 2. Set expectations
        expected_IOError = "Invalid file path extension, should be .json or .geojson."
        expectedTypeError = f"file_path should be string or Path, not {int}"

        # 3. Run test
        with pytest.raises(IOError, match=expected_IOError):
            validate_extension(str(file_path))

        with pytest.raises(TypeError, match=expectedTypeError):
            validate_extension(1)

    @pytest.mark.parametrize("file_name", [("n.json"), ("n.geojson")])
    def test_when_valid_extension_does_not_raise(self, file_name):
        validate_extension(file_name)


class Test_read_mask_output_file:
    def test_when_invalid_file_path_given(self, tmp_path):
        # 1. Set up test data
        not_existing_file = tmp_path / "test.geojson"

        # 2. Set expectation
        import re
        expected_error = f"File path {not_existing_file} not found"
        expected_error_regex = re.escape(expected_error)

        # 3. Run test
        with pytest.raises(FileNotFoundError, match=expected_error_regex):
            read_mask_output_file(not_existing_file)

    def test_when_valid_file_with_no_content_then_returns_expected_geojson(self, tmp_path):
        # 1. Set up test data
        file_path: Path = tmp_path / "empty.geojson"
        with file_path.open("w") as f:
            geojson.dump({}, f)

        # 2. Set up expectations
        expected_error = "File is empty or not a valid geojson file."

        # Run test
        with pytest.raises(IOError, match=expected_error):
            read_mask_output_file(file_path)

    def test_when_valid_file_with_content_then_returns_expected_geojson(self):
        # 1. Set up test data
        file_path = TestUtils.get_local_test_data_dir("maskoutputfile_test_data") / "mask_points.geojson"

        # 2. Read data
        read_geojson = read_mask_output_file(file_path)

        # 3. Verify final expectations
        assert read_geojson, "No geojson data was generated."
        assert isinstance(read_geojson, geojson.FeatureCollection)
        assert read_geojson.is_valid, "The geojson data is not valid."


class Test_write_mask_output_file:
    def test_when_valid_file_path_and_no_mask_point_then_writes_expectations(self, tmp_path: Path):
        # 1. Set up test data
        file_path = tmp_path / "no_mask_points.geojson"

        # 2. Set up expectations
        error_msg = "mask_points cannot be empty"

        # 3. Run test
        with pytest.raises(ValueError, match=error_msg):
            write_mask_output_file(file_path=file_path, mask_points=None)

    def test_when_valid_file_path_and_mask_points_then_writes_expectations(self, tmp_path):
        # 1. Set up test data
        file_path = tmp_path / "mask_points.geojson"
        mask_points = [
            geojson.Feature(geometry=geojson.utils.generate_random("Point")),
            geojson.Feature(geometry=geojson.utils.generate_random("Point")),
        ]

        # 2. Set expectations
        expected_mask_points = geojson.FeatureCollection(mask_points)

        # 3. Run test
        write_mask_output_file(file_path, mask_points)

        # 4. Verify final expectations
        assert file_path.exists(), f"File {file_path} not found."

        with file_path.open("r") as geojson_file:
            read_mask_points = geojson.load(geojson_file)

        assert read_mask_points, f"No mask points were read from {file_path}"
        assert expected_mask_points == read_mask_points, f"Expected {expected_mask_points} , but got {read_mask_points}"
