import unittest
import pytest
import sys
import os
import numbers

from bin.fm2profConsole import cli
from click.testing import CliRunner


class Test_Console:
    @pytest.mark.unittest
    def test_no_flag(self):
        # 1. Set up initial test data
        runner = CliRunner()
        
        # 2. Run test
        try:
            result = runner.invoke(cli, "")
            assert not result.exception
            #self.assertEqual(0, result.exit_code)
            #self.assertIn('Find: 3 sample', result.output)
        except:
            pass
            pytest.fail('No exception expected.')

    def test_new_file(self):
        # 1. Set up initial test data
        runner = CliRunner()
        
        # 2. Run test
        try:
            result = runner.invoke(cli, ["-n EmptyProject.ini"])

            assert not result.exception
        except:
            pass
            pytest.fail('No exception expected.')