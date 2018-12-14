import unittest
import sys, os
import TestUtils

# High level acceptance tests, these are the ones who are only meant to generate output files
# for the testers to verify (in Teamcity) whether the runs generate the expected files or not.

class Main_AcceptanceTests(unittest.TestCase):  
    def __run_main_with_arguments(self, map_file, css_file, chainage_file, output_directory):
        pythonCall = "fm2prof\\main.py -i {0} -i {1} -i {2} -o {3}".format(map_file, css_file, chainage_file, output_directory)
        os.system("python {0}".format(pythonCall))

    def test_runbasecase(self):
        directory = TestUtils.get_test_dir("case_01_rectangle")       
        map_file = directory + 'Data\\FM\\50x25_mesh\\FlowFM_fm2prof_map.nc'
        css_file = directory + 'Data\\cross_section_locations.xyz'
        chainage_file = directory + 'Data\\cross_section_chainages.txt'
        output_directory = directory + 'Output'
        self.__run_main_with_arguments(map_file, css_file, chainage_file, output_directory)