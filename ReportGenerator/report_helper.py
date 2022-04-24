import os

from tests.TestUtils import TestUtils as TestUtils

_case_lab_key = "case_label"
_case_key_key = "case_key"
_case_fig_key = "figures"

_fig_name_key = "figure_name"
_fig_key_key = "figure_key"
_fig_rel_path_key = "figure_rel_path"
_fig_path_key = "figure_path"

__templates_dir = "report_templates"


def _get_template_folder(template_dir: str):
    """Returns a template type within the Templates dir.

    Arguments:
        template_dir {str} -- Type of report template.

    Returns:
        {str} -- Path to report template dir.
    """
    report_helper_dir = os.path.dirname(__file__)
    templates_dir_path = os.path.join(report_helper_dir, __templates_dir)
    return os.path.join(templates_dir_path, template_dir)


def _get_case_figures(case_name: str, case_label: str, data_dir: str):
    """Gets a list of figures dictionary

    Arguments:
        case_name {str} -- Name of the case folder.
        case_label {str} -- How should the case name be displayed.
        data_dir {str} -- Path where the data is stored.

    Returns:
        {list} -- List of dictionaries (figures).
    """
    figures_list = {}
    case_dir = os.path.join(data_dir, case_name)
    case_fig_dir = os.path.join(case_dir, "Figures")
    if not os.path.exists(case_fig_dir):
        return figures_list

    # Create entries for each figure in the directory
    for subdir in os.listdir(case_fig_dir):
        rel_path = "../{}/Figures/{}/".format(case_name, subdir)
        figures_list[subdir] = list()
        for fig in os.listdir(os.path.join(case_fig_dir, subdir)):
            fig_base_name = os.path.split(fig)[1]
            fig_name = case_label + " " + os.path.splitext(fig_base_name)[0]
            fig_name = fig_name.capitalize()
            fig_key = fig_name.replace(" ", "_")
            figure_dict = {
                _fig_name_key: fig_name.replace("_", " "),
                _fig_key_key: fig_key.replace("_", "") + "Key",
                _fig_rel_path_key: rel_path + fig,
                _fig_path_key: os.path.join(case_dir, subdir, fig),
            }
            figures_list[subdir].append(figure_dict)
        figures_list[subdir] = sorted(
            figures_list[subdir], key=lambda fig: fig[_fig_name_key]
        )
    return figures_list


def _get_all_cases_and_figures(scenarios: list, data_dir: str) -> dict:
    """Gets a dictionary of case - figures.

    Arguments:
        scenarios {list} -- List of case scenarios.
        data_dir {str} -- Path where the data is stored.

    Returns:
        {dict} -- Dictionary where keys are case_names
        and values their figure dictionaries.
    """
    cases_dictionary = {}
    for case in scenarios:
        case_base = case.capitalize()
        case_label = case_base.replace("_", " ")
        case_figures = _get_case_figures(case, case_label, data_dir)
        case_dict = {
            _case_lab_key: case_label,
            _case_key_key: case_base.replace("_", "") + "Key",
            _case_fig_key: case_figures,
        }
        cases_dictionary[case] = case_dict

    return cases_dictionary
