import contextlib
import os
from pathlib import Path

import pytest

try:
    from pip import main as pipmain
except Exception as e_info:
    from pip._internal import main as pipmain


class TestUtils:

    _name_external = "external_test_data"
    _name_local = "test_data"
    _name_artifacts = "artifacts"
    _temp_copies = "temp-copies"

    @staticmethod
    def install_package(package: str):
        """Installs a package that is normally only used
        by a test configuration.

        Arguments:
            package {str} -- Name of the PIP package.
        """
        pipmain(["install", package])

    @staticmethod
    def get_external_test_data_dir() -> Path:
        """
        Gets the path to the external test data directory.

        Returns:
            Path: Directory path.
        """
        return Path(__file__).parent / TestUtils._name_external

    @staticmethod
    def get_external_test_data_subdir(subdir: str) -> Path:
        return TestUtils.get_external_test_data_dir() / subdir

    @staticmethod
    def get_artifacts_test_data_dir() -> Path:
        return Path(__file__).parent / TestUtils._name_artifacts

    @staticmethod
    def get_local_test_data_dir(dir_name: str) -> Path:
        """Returns the desired directory relative to the test data.

        Avoiding extra code on the tests.
        """
        return TestUtils.get_test_data_dir(dir_name, TestUtils._name_local)

    @staticmethod
    def get_external_repo(dir_name: str) -> Path:
        """
        Returns the parent directory of this repo directory.

        Args:
            dir_name (str): Repo 'sibbling' of the current one.

        Returns:
            Path: Path to the sibbling repo.
        """
        return Path(__file__).parent.parent.parent / dir_name

    @staticmethod
    def get_test_data_dir(dir_name: str, test_data_name: str) -> Path:
        """
        Returns the desired directory relative to the test external data.
        Avoiding extra code on the tests.
        """
        return Path(__file__).parent / test_data_name / dir_name

    @staticmethod
    def get_local_test_file(filepath: str) -> Path:
        return Path(__file__).parent / TestUtils._name_local / filepath

    @staticmethod
    @contextlib.contextmanager
    def working_directory(path: Path):
        """Changes working directory and returns to previous on exit."""
        prev_cwd = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev_cwd)


skipwhenexternalsmissing = pytest.mark.skipif(
    not (TestUtils.get_external_test_data_dir().is_dir()),
    reason="Only to be run to generate expected data from local machines.",
)
