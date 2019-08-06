import unittest
import pytest
import sys
import os
import numbers

import shutil
import geojson
import json

from fm2prof.MaskPoint import *
from tests.TestUtils import TestUtils
# from fm2prof.IniFile import IniFile

_root_output_dir = None


class Test_MaskPoint:

    @pytest.mark.unittest
    def test_when_no_coordinates_given_then_no_exception_is_risen(self):
        try:
            mask_point = MaskPoint(None, None)
        except:
            pytest.fail('No exception expected')
        assert mask_point is not None, '' + \
            'No MaskPoint object was created'

    @pytest.mark.unittest
    def test_when_no_added_dict_given_then_extend_properties_does_not_rise(
            self):
        # 1. Set up test model
        extended_dict = None
        mask_point = MaskPoint(None, None)

        # 2. Verify initial expectations
        assert mask_point is not None

        # 3. Run test
        try:
            mask_point.extend_properties(extended_dict)
        except:
            pytest.fail('No exception expected')

    @pytest.mark.unit
    def test_when_dict_given_then_extend_properties_does_not_rise(
            self):
        # 1. Set up test model
        extended_dict = {
            'dummyKey': 'dummyValue'
            }
        mask_point = MaskPoint(None, None)

        # 2. Verify initial expectations
        assert mask_point is not None

        # 3. Run test
        try:
            mask_point.extend_properties(extended_dict)
        except:
            pytest.fail('No exception expected')

    @pytest.mark.integration
    def test_when_coordinates_given_then_dump_geodata_contains_them(self):
        # 1. Set up test model
        coord_x = 4.2
        coord_y = 2.4
        mask_point = MaskPoint(coord_x, coord_y)

        # 2. Set up expectations
        expected_data = {
            'type': 'Point',
            'coordinates': [coord_x, coord_y],
            'properties': {},
        }

        # 2. Dump data
        dump_data = geojson.dumps(mask_point, sort_keys=True)

        # 3. Verify the data matches the expectatiosn
        assert dump_data is not None

        # 4. Load dumped data
        json_dump_data = json.loads(dump_data)
        assert json_dump_data is not None
        assert json_dump_data == expected_data, '' + \
            'Expected data {}, but was {}'.format(expected_data, dump_data)
