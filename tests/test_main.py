import unittest
import sys, os
from fm2prof import main
import TestUtils

class SystemTests(unittest.TestCase):
    def f(self):
        return 4
    
    def test_mytest_should_pass(self):
        assert self.f() == 4
    
    def test_mytest_should_fail(self):
        assert self.f() == 3
    
    def test_runbasecase(self):
        directory = TestUtils.get_test_dir("case_01_rectangle")       
        map_file = directory + '\\Data\\FM\\50x25_mesh\\FlowFM_fm2prof_map.nc'
        css_file = directory + '\\Data\\cross_section_locations.xyz'
        chainage_file = directory + '\\Data\\cross_section_chainages.txt'
        main.runfile(directory, map_file,css_file, chainage_file)