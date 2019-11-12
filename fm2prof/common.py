"""
Copyright (C) Stichting Deltares 2019. All rights reserved.

This file is part of the Fm2Prof.

The Fm2Prof is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

All names, logos, and references to "Deltares" are registered trademarks of
Stichting Deltares and remain full property of Stichting Deltares at all times.
All rights reserved.
"""

import re
from collections import OrderedDict

def read_deltares_ini(filename):
    """
    lala

    :param filename:
    :return:
    """
    section_re = re.compile(r'^\s*\[\s*(.+?)\s*\]')
    keyval_re  = re.compile(r'^\s*(.+?)\s*=\s*(.+?)\s*$')
    ini = []
    with open(filename, 'r') as f:
        for line in f:
            section = section_re.search(line)
            if section:
                sectionname = section.group(1)
                ini.append({'SectionName': sectionname})

            keyvalue = keyval_re.search(line)
            if keyvalue:
                key = keyvalue.group(1)
                value = keyvalue.group(2)
                ini[-1][key] = value

    return ini


class MDF():
    def __init__(self):
        self.keywords = OrderedDict()

    def read(self, mdf_file):
        """
        Read Delft3D-Flow *.mdf file into dictionary.
        - Preserves order of original mdf
        - Preserves comments

        mdf_file : path to an *.mdf file
        """

        keywords = OrderedDict()
        comment_counter = 1
        with open(mdf_file, 'r') as f:
            for line in f.readlines():
                if '=' in line:
                    keyword, value = line.split('=', 1)
                    keyword = keyword.strip()
                    if keyword == 'Commnt':
                        keyword += '_%003i' % comment_counter
                        comment_counter += 1
                        keywords[keyword] = value.strip()
                    else:
                        keywords[keyword] = _RHS2val_(value.strip())
        self.keywords = keywords



    def write(self, output_file):
        """
        Write to *.mdf file

        SYNTAX
        output_file: path to output file
        """
        if not self.keywords:
            raise ValueError('No data to write')
        else:

            with open(output_file, 'w') as f:
                for key in self.keywords:
                    f.write(_val2RHS_(key, self.keywords[key]))
                f.close()



def _RHS2val_(line):
    """parse 1(!) RHS line value from *.mdf file to a str, '' or float"""

    if '#' in line:
        dummy, value, dummy = line.split('#', 2)
    else:
        value = line.strip()
        try:
            value = map(float, value.split())
        except:
            print(value)
            raise ValueError
    return value

def _val2RHS_(keyword, value):
    """
    parse a list of str, '' or floats to multiple (!, if needed) RHS *.mdf lines
    NOTE: Delft3D will through random error if area before '=' (containing keyword) is not exactly 7 characters. Hence
    the 'keyword.ljust(7)'
    """

    # values that need to be written column-wise rather than row wise (although short row vectors are allowed)
    columnwise_list = ['Thick', 'Rettis', 'Rettib', 'u0', 'v0', 's0', 't0', 'c01', 'c02', 'c03', 'c04', 'c01']
    if keyword.split('_')[0] == 'Commnt':
        keyword = 'Commnt'
    if  type(value) is str:
        if keyword=='Runtxt':
            MaximumWidth = 30
            lineOut = '%s= #%s#\n' % (keyword.ljust(7), value[:MaximumWidth])
            for i in range(MaximumWidth, len(value), MaximumWidth):
                lineOut += '         #%s#\n' % value[i:i+MaximumWidth]
        elif keyword == 'Commnt':
            lineOut = '%s= %s\n' % (keyword.ljust(7), value)
        else:
            lineOut = '%s= #%s#\n' % (keyword.ljust(7), value)
    else:
        if keyword in columnwise_list:
            lineOut = '%s= %g\n' % (keyword.ljust(7), value[0])
            for val in value[1:]:
                lineOut += '         %g\n' % val
        else:
         lineOut = '%s= %s\n' % (keyword.ljust(7), ' '.join(("%g" % x) for x in value))
    return lineOut


def get_color_palette(palette='tableau20'):
    """

    :param palette: name of an available palette. Available:
                    'tableau20': colorset of 20 colors
                    'tableau10': colorset of 10 colors
                    'tableau10_light': colorset of 10 lighter colors
                    'tableau10_medium': colorset of 10 colors of medium intensity
                    'color_blind_10': colorset of 10 colors for color blindness
    :return: list of tuples. Each tuple represent a rgb color (values between 0 and 1)
    """
    color_dict = dict()
    palette = palette.lower()
    # Tableau colorsets. Tableau is a (free) service for web publishing of interactive data.
    # See https://public.tableau.com
    color_dict['tableau20'] = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                               (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                               (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                               (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                               (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    color_dict['tableau10'] = [color_dict['tableau20'][i] for i in range(0, 20, 2)]
    color_dict['tableau10_light'] = [color_dict['tableau20'][i] for i in range(1, 20, 2)]
    color_dict['tableau10_medium'] = [(114, 158, 206), (255, 158, 74), (103, 191, 92), (237, 102, 93),
                                      (173, 139, 201), (168, 120, 110), (237, 151, 202), (162, 162, 162),
                                      (205, 204, 93), (109, 204, 218)]
    color_dict['color_blind_10'] = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),
                                    (95, 158, 209), (200, 82, 0), (137, 137, 137), (162, 200, 236),
                                    (255, 188, 121), (207, 207, 207)]

    color_out = list()
    for color in (color_dict[palette]):
        r, g, b = color
        color_out.append((r / 255., g / 255., b / 255.))

    return color_out