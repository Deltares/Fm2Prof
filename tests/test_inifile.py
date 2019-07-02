import unittest, pytest
import sys, os

import shutil
import TestUtils

from fm2prof.main import IniFile

_root_output_dir = None

@pytest.mark.unittest
def test_IniFile_When_No_FilePath_Then_No_Exception_Is_Risen():
    """ 1. Set up initial test data """
    iniFilePath = ''

    """ 2. Run test """
    try:
        iniFile = IniFile(iniFilePath)
    except:
        pytest.fail('No exception expected.')    

@pytest.mark.unittest
def test_IniFile_ReadIniFile_When_No_FilePath_Then_No_Exception_Is_Risen():
    """ 1. Set up initial test data """
    iniFilePath = ''
    iniFile = IniFile(iniFilePath)

    """ 2. Run test """
    with pytest.raises(Exception) as e_info:
        iniFile._read_inifile(iniFilePath)