import os 
import json
from shapely.geometry import shape, GeometryCollection, Point
import logging
import numpy as np 

class RegionPolygonFile:
    
    def __init__(self, region_file_path, logger):
        self.regions = dict()

        self.set_logger(logger)
        self.read_region_file(region_file_path)

    def read_region_file(self, file_path):
        RegionPolygonFile._validate_extension(file_path)

        with open(file_path) as geojson_file:
            geojson_data = json.load(geojson_file).get("features")
        
        regions = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in geojson_data])
        region_names = [feature.get('properties').get('Name') for feature in geojson_data]

        for region, region_name in zip(regions, region_names):
            self.regions[region_name] = region
        
        self._validate_regions()

    def classify_points(self, points):
        """ 
        Classifies points as belonging to which region

        Points = list of tuples [(x,y), (x,y)]
        """
        
        # Convert to shapely point
        points = [Point(xy) for xy in points]
        points_regions = ['undefined']*len(points)
        
        # Assign point to region
        for i, point in enumerate(points):
            for region in self.regions:
                if point.within(self.regions.get(region)):
                    points_regions[i] = region
                    break
        
        return np.array(points_regions)

    def _validate_regions(self):
        self._set_logger_message("Validating Region file")
        
        number_of_regions = len(self.regions)
        self._set_logger_message("{} regions found: {}".format(number_of_regions,
                                  list(self.regions.keys())
                                  ))

        # Test if polygons overlap
        for region_name, region in self.regions.items():
            for test_name, test_region in self.regions.items():
                if region_name == test_name:
                    # region will obviously overlap with itself
                    pass
                else:
                    if region.intersects(test_region):
                        self._set_logger_message("{} overlaps {}.".format(region_name, test_name),
                                                 level= 'warning')


    @staticmethod
    def _validate_extension(file_path: str):
        if (not file_path):
            # Should not be evaluated
            return
        if (not file_path.endswith('.json') and
                not file_path.endswith('.geojson')):
            raise IOError(
                'Invalid file path extension, ' +
                'should be .json or .geojson.')

    def set_logger(self, logger):
        """ should be given a logger object (python standard library) """
        assert isinstance(logger, logging.Logger), '' + \
            'logger should be instance of logging.Logger class'

        self.__logger = logger

    def _set_logger_message(self, err_mssg: str, level='info'):
        """Sets message to logger if this is set.

        Arguments:
            err_mssg {str} -- Error message to send to logger.
        """
        if not self.__logger:
            return

        if level.lower() not in [
                'info', 'debug', 'warning', 'error', 'critical']:
                self.__logger.error(
                    "{} is not valid logging level.".format(level.lower()))

        if level.lower() == 'info':
            self.__logger.info(err_mssg)
        elif level.lower() == 'debug':
            self.__logger.debug(err_mssg)
        elif level.lower() == 'warning':
            self.__logger.warning(err_mssg)
        elif level.lower() == 'error':
            self.__logger.error(err_mssg)
        elif level.lower() == 'critical':
            self.__logger.critical(err_mssg)