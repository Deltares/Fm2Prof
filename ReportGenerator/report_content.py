from typing import List, Set, Dict, Tuple, Optional
from abc import ABC, abstractmethod

import ReportGenerator.report_helper as ReportHelper


class ReportContent(ABC):

    def __init__(
            self,
            scenarios_ids: List[str] = None,
            data_dir: str = None):
        """Creates a report content with optional parameters. The concrete
        classes can define all the needed information to generate reports.

        Keyword Arguments:
            scenarios_ids {List[str]}
                -- List of scenarios that need reporting (default: {None}).
            data_dir {str} -- Location of figures to report (default: {None}).
        """
        self.scenarios_ids = scenarios_ids
        self.data_dir = data_dir

    @property
    def cases_and_figures(self) -> dict:
        return ReportHelper._get_all_cases_and_figures(
                scenarios=self.scenarios_ids,
                data_dir=self.data_dir)

    @property
    @abstractmethod
    def project_name(self) -> str:
        raise NotImplementedError('Implement in child class.')

    @property
    @abstractmethod
    def project_version(self) -> str:
        raise NotImplementedError('Implement in child class.')

    @property
    @abstractmethod
    def project_number(self) -> str:
        raise NotImplementedError('Implement in child class.')

    @property
    @abstractmethod
    def authors_list(self) -> List[str]:
        raise NotImplementedError('Implement in child class.')

    @property
    @abstractmethod
    def case_description_dict(self) -> Dict[str, str]:
        """Returns a dictionary with the case name and its description.

        Raises:
            NotImplementedError: If not implemented in child class.

        Returns:
            Dict[str, str] -- Dictionary with case's description.
        """
        raise NotImplementedError('Implement in child class.')
