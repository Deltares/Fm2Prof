import unittest
import pytest
import sys
import os
import numbers

import shutil
import json
import geojson


from fm2prof.MaskOutputFile import MaskOutputFile
from tests.TestUtils import TestUtils


class MaskOutputFileHelper:

    @staticmethod
    def get_mask_point():
        mask_point = geojson.utils.generate_random('Point')
        return mask_point


class Test_create_mask_point:

    @pytest.mark.unittest
    def test_when_no_coords_does_not_raise(self):
        # 1. Set up test model
        coords = None
        properties = None
        mask_point = None
        # 2. Set up initial expectations

        # 3. Do test
        try:
            mask_point = MaskOutputFile.create_mask_point(coords, properties)
        except:
            pytest.fail('Exception risen but not expected.')
        # 4. Verify final expectations
        assert mask_point is not None, '' + \
            'No mask_point generated'

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "coords_values",
        [(4.2, 2.4), (4.2, 2.4, 42)])
    def test_when_valid_coords_does_not_raise(self, coords_values: tuple):
        # 1. Set up test model
        coords = coords_values
        properties = None
        mask_point = None

        # 2. Do test
        try:
            mask_point = MaskOutputFile.create_mask_point(coords, properties)
        except:
            pytest.fail('Exception risen but not expected.')
        # 3. Verify final expectations
        assert mask_point is not None, '' + \
            'No mask_point generated'

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "coords_values",
        [(4.2)])
    def test_when_invalid_coords_raises(self, coords_values: tuple):
        # 1. Set up test model
        coords = coords_values
        properties = None
        mask_point = None
        exception_thrown = False
        # 2. Do test
        try:
            mask_point = MaskOutputFile.create_mask_point(coords, properties)
        except:
            exception_thrown = True

        # 3. Verify final expectations
        assert exception_thrown, '' + \
            'No exception was thrown but it was expected.'
        assert mask_point is None, '' + \
            'Mask point generated but was not expected.'

    @pytest.mark.unittest
    def test_when_no_properties_does_not_raise(self):
        # 1. Set up test model
        coords = (4.2, 2.4)
        properties = None
        mask_point = None
        # 2. Set up initial expectations

        # 3. Do test
        try:
            mask_point = MaskOutputFile.create_mask_point(coords, properties)
        except:
            pytest.fail('Exception risen but not expected.')
        # 4. Verify final expectations
        assert mask_point is not None, '' + \
            'No mask_point generated'


class Test_validate_extension:

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "file_name",
        [(None), ('')])
    def test_when_no_file_path_doesnot_raise(self, file_name):
        try:
            MaskOutputFile.validate_extension(file_name)
        except:
            pytest.fail('Exception risen but not expected.')

    @pytest.mark.unittest
    def test_when_invalid_extension_raises_expected_exception(self):
        # 1. Set up test data
        file_name = 'test_file.wrongextension'

        # 2. Set expectations
        expected_error = '' + \
            'Invalid file path extension, should be .json or .geojson.'

        # 3. Run test
        with pytest.raises(IOError) as e_info:
            MaskOutputFile.validate_extension(file_name)

        # 4. Verify final expectations
        error_message = str(e_info.value)
        assert error_message == expected_error, '' + \
            'Expected exception message {},'.format(expected_error) + \
            'retrieved {}'.format(error_message)

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "file_name",
        [('n.json'), ('n.geojson')])
    def test_when_valid_extension_does_not_raise(self, file_name):
        try:
            MaskOutputFile.validate_extension(file_name)
        except:
            pytest.fail('Exception risen but not expected.')


class Test_read_mask_output_file:

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "file_path_name",
        [
            (None),
            ('invalidfile.geojson'),
            ('invalidfile.json'),
            ('invalidfile.xyz')])
    def test_when_invalid_file_path_given_then_emtpy_geojson(
            self, file_path_name):
        # 1. Set up test data
        read_geojson = None

        # 2. Set up initial expectations
        expected_geojson = geojson.FeatureCollection(None)

        # 3. Read data
        try:
            read_geojson = MaskOutputFile.read_mask_output_file(file_path_name)
        except:
            pytest.fail('Exception thrown but not expected.')

        # 4. Verify final expectations
        assert read_geojson, '' + \
            'No geojson data was generated.'
        assert read_geojson == expected_geojson, '' + \
            'Expected {} but got {}'.format(expected_geojson, read_geojson)

    @pytest.mark.integrationtest
    def test_when_file_path_wrong_extension_then_raise_exception(self):
        # 1. Set up test data
        test_dir = TestUtils.get_local_test_data_dir(
            'maskoutputfile_test_data')
        file_name = 'test_file.wrongextension'
        file_path = os.path.join(test_dir, file_name)
        read_geojson_data = None

        # 2. Set expectations
        assert os.path.exists(file_path), '' + \
            'File {} could not be found.'.format(file_path)
        expected_error = '' + \
            'Invalid file path extension, should be .json or .geojson.'

        # 3. Run test
        with pytest.raises(IOError) as e_info:
            read_geojson_data = MaskOutputFile.read_mask_output_file(file_path)

        # 4. Verify final expectations
        assert not read_geojson_data
        error_message = str(e_info.value)
        assert error_message == expected_error, '' + \
            'Expected exception message {},'.format(expected_error) + \
            'retrieved {}'.format(error_message)

    @pytest.mark.systemtest
    @pytest.mark.parametrize(
        "file_name",
        [('no_mask_points.geojson'), ('no_content.geojson')])
    def test_when_valid_file_with_no_content_then_returns_expected_geojson(
            self, file_name):
        # 1. Set up test data
        test_dir = TestUtils.get_local_test_data_dir(
            'maskoutputfile_test_data')
        file_path = os.path.join(test_dir, file_name)

        # 2. Set up initial expectations
        expected_geojson = geojson.FeatureCollection(None)
        assert os.path.exists(file_path), '' + \
            'File not found at {}'.format(file_path)

        # 2. Read data
        try:
            read_geojson = MaskOutputFile.read_mask_output_file(file_path)
        except:
            pytest.fail('Exception thrown but not expected.')

        # 3. Verify final expectations
        assert read_geojson, '' + \
            'No geojson data was generated.'
        assert read_geojson == expected_geojson, '' + \
            'Expected {} but got {}'.format(expected_geojson, read_geojson)

    @pytest.mark.systemtest
    def test_when_valid_file_with_content_then_returns_expected_geojson(
            self):
        # 1. Set up test data
        test_dir = TestUtils.get_local_test_data_dir(
            'maskoutputfile_test_data')
        file_name = 'mask_points.geojson'
        file_path = os.path.join(test_dir, file_name)

        # 2. Verify initial expectations
        assert os.path.exists(file_path), '' + \
            'File not found at {}'.format(file_path)

        # 2. Read data
        try:
            read_geojson = MaskOutputFile.read_mask_output_file(file_path)
        except:
            pytest.fail('Exception thrown but not expected.')

        # 3. Verify final expectations
        assert read_geojson, '' + \
            'No geojson data was generated.'
        assert read_geojson.is_valid, '' + \
            'The geojson data is not valid.'


class Test_write_mask_output_file:

    @pytest.fixture(scope='class')
    def test_folder(self):
        """Prepares the class properties to be used in the tests.
        """
        test_dir = TestUtils.get_test_dir('Output\\WriteMaskOutputFile')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        yield test_dir

    @pytest.mark.unittest
    def test_when_no_file_path_given_then_exception_not_risen(self):
        # 1. Set up test data
        file_path = None
        mask_points = None
        # 2. Verify test
        try:
            MaskOutputFile.write_mask_output_file(file_path, mask_points)
        except:
            pytest.fail('Exception thrown but not expected.')

    @pytest.mark.integrationtest
    def test_when_file_path_with_wrong_extension_then_exception_is_risen(self):
        # 1. Set up test data
        file_path = 'test_file.wrongextension'
        mask_points = None

        # 2. Set expectations
        expected_error = '' + \
            'Invalid file path extension, should be .json or .geojson.'

        # 3. Run test
        with pytest.raises(IOError) as e_info:
            MaskOutputFile.write_mask_output_file(file_path, mask_points)

        # 4. Verify final expectations
        error_message = str(e_info.value)
        assert error_message == expected_error, '' + \
            'Expected exception message {},'.format(expected_error) + \
            'retrieved {}'.format(error_message)

    @pytest.mark.systemtest
    def test_when_valid_file_path_and_no_mask_point_then_writes_expectations(
            self, test_folder):
        # 1. Set up test data
        file_name = 'no_mask_points.geojson'
        file_path = os.path.join(test_folder, file_name)
        mask_points = None

        # 2. Set expectations
        expected_mask_points = geojson.FeatureCollection(mask_points)
        read_mask_points = None

        # 3. Run test
        try:
            MaskOutputFile.write_mask_output_file(file_path, mask_points)
        except:
            pytest.fail('Exception thrown but not expected.')

        # 4. Verify final expectations
        assert os.path.exists(file_path), '' + \
            'File {} not found.'.format(file_path)

        with open(file_path) as geojson_file:
            read_mask_points = geojson.load(geojson_file)

        assert read_mask_points, '' + \
            'No mask points were read from {}'.format(file_path)
        assert expected_mask_points == read_mask_points, '' + \
            'Expected {} ,'.format(expected_mask_points) + \
            'but got {}'.format(read_mask_points)

    @pytest.mark.systemtest
    def test_when_valid_file_path_and_mask_points_then_writes_expectations(
            self, test_folder):
        # 1. Set up test data
        file_name = 'mask_points.geojson'
        file_path = os.path.join(test_folder, file_name)
        mask_points = [
            geojson.Feature(
                geometry=MaskOutputFileHelper.get_mask_point()),
            geojson.Feature(
                geometry=MaskOutputFileHelper.get_mask_point())
        ]

        # 2. Set expectations
        expected_mask_points = geojson.FeatureCollection(mask_points)
        read_mask_points = None

        # 3. Run test
        try:
            MaskOutputFile.write_mask_output_file(file_path, mask_points)
        except:
            pytest.fail('Exception thrown but not expected.')

        # 4. Verify final expectations
        assert os.path.exists(file_path), '' + \
            'File {} not found.'.format(file_path)

        with open(file_path) as geojson_file:
            read_mask_points = geojson.load(geojson_file)

        assert read_mask_points, '' + \
            'No mask points were read from {}'.format(file_path)
        assert expected_mask_points == read_mask_points, '' + \
            'Expected {} ,'.format(expected_mask_points) + \
            'but got {}'.format(read_mask_points)