import os

from fm2prof.MaskPoint import *

import geojson


class MaskOutputFile:
    __geojson_data = None
    __mask_points = []

    @property
    def mask_points(self):
        return self.__mask_points

    @staticmethod
    def get_geojson_feature_collection(points):
        pass

    def read_mask_output_file(self, file_path: str):
        """Imports a GeoJson from a given json file path.

        Arguments:
            file_path {str} -- Location of the json file
        """
        if not file_path or not os.path.exists(file_path):
            return

        with open(file_path) as json_file:
            self.__geojson_data = geojson.load(json_file)

        return self.__geojson_data

    def write_mask_output_file(self, file_path: str):
        pass

    def create_mask_point(self, mask_properties):
        new_mask_point = MaskPoint(None, None)
        new_mask_point.extend_properties(mask_properties)
        return new_mask_point
