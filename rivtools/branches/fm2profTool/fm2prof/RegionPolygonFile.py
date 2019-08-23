import os 
import json
from shapely.geometry import shape, GeometryCollection, Point
import logging
import numpy as np 
from collections import namedtuple

Polygon = namedtuple('Polygon', ['geometry', 'properties'])

class PolygonFile:
    __logger = None

    def __init__(self, logger):
        self.set_logger(logger)
        self.polygons = list()
        self.undefined = 'undefined'
    def classify_points_with_property(self, points, property='name'):
        """ 
        Classifies points as belonging to which region

        Points = list of tuples [(x,y), (x,y)]
        """
        
        # Convert to shapely point
        points = [Point(xy) for xy in points]
        points_regions = [self.undefined]*len(points)
        
        # Assign point to region
        for i, point in enumerate(points):
            for polygon in self.polygons:
                if point.within(polygon.geometry):
                    points_regions[i] = polygon.properties.get(property)
                    break

        return np.array(points_regions)

    def set_logger(self, logger):
        """ should be given a logger object (python standard library) """
        assert isinstance(logger, logging.Logger), '' + \
            'logger should be instance of logging.Logger class'

        self.__logger = logger
    
    def parse_geojson_file(self, file_path):
        """ Read data from geojson file """
        PolygonFile._validate_extension(file_path)

        with open(file_path) as geojson_file:
            geojson_data = json.load(geojson_file).get("features")
        
        for feature in geojson_data:
            feature_props = {k.lower(): v for k, v in feature.get('properties').items()}
            self.polygons.append(Polygon(geometry=shape(feature["geometry"]).buffer(0),
                                         properties=feature_props))
        #polygons = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in geojson_data])
        #polygon_names = [feature.get('properties').get('Name') for feature in geojson_data]

        #for polygon, polygon_name in zip(polygons, polygon_names):
        #    self.polygons[polygon_name] = polygon


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

    def _check_overlap(self):
        for polygon in self.polygons:
            for testpoly in self.polygons:
                if polygon.properties.get('name') == testpoly.properties.get('name'):
                    # polygon will obviously overlap with itself
                    pass
                else:
                    if polygon.geometry.intersects(testpoly.geometry):
                        self._set_logger_message("{} overlaps {}.".format(polygon.properties.get('name'), 
                                                                          testpoly.properties.get('name')),
                                                 level= 'warning')

class RegionPolygonFile(PolygonFile):
    
    def __init__(self, region_file_path, logger):
        super().__init__(logger)
        self.read_region_file(region_file_path)

    @property
    def regions(self):
        return self.polygons

    def read_region_file(self, file_path):
        self.parse_geojson_file(file_path)
        self._validate_regions()

    def _validate_regions(self):
        self._set_logger_message("Validating Region file")
        
        number_of_regions = len(self.regions)
        
        self._set_logger_message("{} regions found".format(number_of_regions))

        # Test if polygons overlap
        self._check_overlap()

    def classify_points(self, points):
        return self.classify_points_with_property(points, property='name')

class SectionPolygonFile(PolygonFile):
    def __init__(self, section_file_path, logger):
        super().__init__(logger)
        self.read_section_file(section_file_path)
        self.undefined = 'main'
    @property
    def sections(self):
        return self.polygons

    def read_section_file(self, file_path):
        self.parse_geojson_file(file_path)
        self._validate_sections()

    def classify_points(self, points):
        return self.classify_points_with_property(points, property='section')


    def _validate_sections(self):
        self._set_logger_message("Validating Section file")
        raise_exception = False
        
        valid_section_keys = {'main', 'floodplain1', 'floodplain2'}
        map_section_keys = {'1':'main',
                            '2': 'floodplain1',
                            '3': 'floodplain2',
                            }

        # each polygon must have a 'section' property
        for section in self.sections:
            if 'section' not in section.properties:
                raise_exception = True
                self._set_logger_message('Polygon {} has no property "section"'.format(section.properties.get('name')),
                                          level='error')
            section_key = str(section.properties.get('section'))
            if section_key not in valid_section_keys:
                if section_key not in list(map_section_keys.keys()):
                    raise_exception = True
                    self._set_logger_message('{} is not a recognized section'.format(section_key), 
                    level='error')
                else:
                    self._set_logger_message('remapped section {} to {}'.format(section_key, 
                                                                                map_section_keys[section_key]),
                    level='warning')
                    section.properties[section_key] = map_section_keys.get(section_key)
        # check for overlap (only raise a warning)
        self._check_overlap()

        if raise_exception:
            raise AssertionError('Section file could not validated')
        else:
            self._set_logger_message('Section file succesfully validated')