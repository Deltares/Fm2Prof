import os
import geojson


class MaskOutputFile:

    @staticmethod
    def create_mask_point(coords: geojson.coords, properties: dict):
        """Creates a Point based on the properties and coordinates given.

        Arguments:
            coords {geojson.coords} --
                Coordinates tuple (x,y) for the mask point.
            properties {dict} -- Dictionary of properties
        """
        output_mask = geojson.Point(coords)
        if properties:
            output_mask.properties = properties
        return output_mask

    @staticmethod
    def validate_extension(file_path: str):
        if (not file_path):
            # Should not be evaluated
            return
        if (not file_path.endswith('.json') and
                not file_path.endswith('.geojson')):
            raise IOError(
                'Invalid file path extension, ' +
                'should be .json or .geojson.')

    @staticmethod
    def read_mask_output_file(file_path: str):
        """Imports a GeoJson from a given json file path.

        Arguments:
            file_path {str} -- Location of the json file
        """
        geojson_data = geojson.FeatureCollection(None)
        if not file_path or not os.path.exists(file_path):
            return geojson_data

        MaskOutputFile.validate_extension(file_path)
        with open(file_path) as geojson_file:
            try:
                geojson_data = geojson.load(geojson_file)
            except:
                return geojson_data

        return geojson_data

    @staticmethod
    def write_mask_output_file(file_path: str, mask_points: list):
        """Writes a .geojson file with a Feature collection containing
        the mask_points list given as input.

        Arguments:
            file_path {str} -- file_path where to store the geojson.
            mask_points {list} -- List of features to output.
        """
        if file_path:
            MaskOutputFile.validate_extension(file_path)
            feature_collection = geojson.FeatureCollection(mask_points)
            with open(file_path, 'w') as f:
                geojson.dump(
                    feature_collection,
                    f,
                    indent=4)
