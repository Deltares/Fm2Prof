import unittest, pytest
import sys, os
import TestUtils

class Main_UnitTests(unittest.TestCase):
    def f(self):
        return 4
    
    @pytest.mark.unittest
    def test_mytest_should_pass(self):
        assert self.f() == 4
    
    @pytest.mark.unittest
    def test_mytest_should_fail(self):
        assert self.f() == 3