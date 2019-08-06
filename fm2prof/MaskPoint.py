import geojson


class MaskPoint:

    __x = None
    __y = None
    __properties = {}

    def __init__(self, x: float, y: float):
        self.__x = x
        self.__y = y

    def extend_properties(self, added_dict: dict):
        """Merge a new set of properties with the existent ones.

        Arguments:
            added_dict {dict} -- New properties to add to the point.
        """
        if added_dict:
            self.__properties = {**self.__properties, **added_dict}

    @property
    def __coordinates(self):
        return (self.__x, self.__y)

    @property
    def __geo_interface__(self):
        return {
            'type': 'Point',
            'coordinates': self.__coordinates,
            'properties': self.__properties,
            }
