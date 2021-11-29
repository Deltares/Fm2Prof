import os
import shutil
from typing import List, Set, Dict, Tuple, Optional

import ReportGenerator.report_helper as ReportHelper
from ReportGenerator.report_content import ReportContent


class LatexReport:

    __latex_template_dir = 'latex_report'
    __latex_template_name = 'report_template.tex'
    __latex_report_name = 'acceptance_report'

    def __init__(self, report_content: ReportContent):
        self.report_content = report_content

    @property
    def __replace_map_dict(self) -> dict:
        if not self.report_content:
            raise ValueError(
                'Report content should have been set.')
        return {
            '%ProgramName%': self.report_content.project_name,
            '%ProgramVersion%': self.report_content.project_version,
            '%ProjectNumber%': self.report_content.project_number,
            '%AUTHORS%': self.__generate_python_authors(),
            '%CONTENT%': self.__generate_python_chapters(),
        }

    @staticmethod
    def convert_to_pdf(report_path: str):
        """Given a path to the LaTeX file, its PDF is generated
        through system calls

        Arguments:
            report_path {str} -- Path to LaTeX file
        """
        report_dir = os.path.dirname(report_path)
        current_wc = os.getcwd()
        os.chdir('{}'.format(report_dir))
        try:
            pycall = 'pdflatex {}/{}.tex'.format(
                os.path.join(current_wc, report_dir),
                LatexReport.__latex_report_name)
            os.system('{} > ar_first_Log.txt'.format(pycall))
            os.system('{} > ar_first_toc_Log.txt'.format(pycall))
            os.system('{} > ar_last_Log.txt'.format(pycall))
            # os.system('buildDoc.bat')
        except Exception as e_info:
            print('Error while generating pdf: {}'.format(e_info))

        os.chdir('{}'.format(current_wc))
        pdf_name = LatexReport.__latex_report_name + '.pdf'
        pdf_path = os.path.join(report_dir, pdf_name)

        if not os.path.exists(pdf_path):
            raise Exception('PDF File was not generated.')

    def generate_latex_report(self, target_dir: str):
        """Generates a .tex from a template with content that is formated
        into strings..
        Templates are copied from a source directory into a target (report)
        directory.

        Arguments:
            target_dir {str} -- Target directory where to generate report.
            report_dict {dict} -- Dictionary with preformated strings.

        Returns:
            {str} -- Path to the new LaTeX file.
        """
        latex_report_path = \
            self.__get_latex_report_file_path(target_dir=target_dir)

        # Read template content.
        with open(latex_report_path, encoding='UTF-8') as f:
            file_str = f.read()

        # Replace template content.
        replaced_report = \
            self.__replace_latex_template_content(
                report_template=file_str)

        # Add new content to LaTeX report.
        with open(latex_report_path, 'w', encoding='UTF-8') as f:
            f.write(replaced_report)

        if not os.path.exists(latex_report_path):
            error_mssg = '' + \
                'Latex report could not be found on path ' + \
                '{}.'.format(latex_report_path)
            raise FileNotFoundError(error_mssg)

        return latex_report_path

    def __get_latex_report_file_path(self, target_dir: str) -> str:
        # Get Template directory.
        template_dir = ReportHelper._get_template_folder(
            self.__latex_template_dir)
        if not os.path.exists(template_dir):
            raise IOError(
                'Template directory could not be found at' +
                ' {}'.format(template_dir))

        # Clean the report dir (target)
        report_dir = os.path.join(target_dir, self.__latex_template_dir)
        if os.path.exists(report_dir):
            shutil.rmtree(report_dir)

        # Copy new templates.
        shutil.copytree(template_dir, report_dir, False, None)
        template_path = os.path.join(report_dir, self.__latex_template_name)
        if not os.path.exists(template_path):
            error_mssg = '' + \
                'Latex template could not be found on path ' + \
                '{}.'.format(template_path)
            raise FileNotFoundError(error_mssg)

        # Generate LaTeX report file.
        latex_report = self.__latex_report_name + '.tex'
        latex_report_path = os.path.join(report_dir, latex_report)
        shutil.move(template_path, latex_report_path)
        if not os.path.exists(latex_report_path):
            error_mssg = '' + \
                'Latex report could not be found on path ' + \
                '{}.'.format(latex_report_path)
            raise FileNotFoundError(error_mssg)
        return latex_report_path

    def __replace_latex_template_content(
            self, report_template: str) -> str:
        """Returns a latex template with the replaced content.

        Arguments:
            report_template {str} -- Latex template as a string.
            report_content_dict {dict} -- Dictionary of content to inject.

        Returns:
            str -- Replaced content.
        """

        replaced_report = report_template
        for key, replace_value in self.__replace_map_dict.items():
            replaced_report = replaced_report.replace(key, replace_value)
        return replaced_report

    def __generate_python_authors(self) -> str:
        """Generates a LaTeX string containing a list of authors.

        Returns:
            {str} -- LaTeX string containing authors.
        """
        author_template = \
            '\\author{author_pos}{{{author_name}}}\n'
        authors_content = ''
        for pos, author in enumerate(self.report_content.authors_list):
            authors_content += \
                author_template.format(
                    author_pos='i'*(pos+1),
                    author_name=author)
        return authors_content

    def __generate_python_chapters(self):
        """Generates a LaTeX string containing a chapter per case.

        Raises:
            ValueError -- When no content is generated from the dictionary.

        Returns:
            {str} -- LaTeX string containing chapter and figures.
        """
        cases_dict = self.report_content.cases_and_figures
        if not cases_dict:
            error_mssg = '' + \
                'No LaTeX content was generated from, ' + \
                'because no data was found'
            raise ValueError(error_mssg)

        temp_name_key = 'CASE_NAME'
        temp_key_key = 'CASE_KEY'
        chapter_template = \
            '\\chapter{{{temp_name_key}}}\n' + \
            '\\label{{{temp_key_key}}}\n'
        case_description_template = \
            '\\section{{Case description}}' + \
            '\n{case_description}\n'
        chapters_content = ''
        for case in sorted(cases_dict):
            case_dict = cases_dict[case]
            # Get values from dictionary.
            case_name = case_dict.get(ReportHelper._case_lab_key)
            case_key = case_dict.get(ReportHelper._case_key_key)
            case_figs = case_dict.get(ReportHelper._case_fig_key)

            # Input Chapter header
            chapters_content += \
                chapter_template.format(
                    temp_name_key=case_name,
                    temp_key_key=case_key)

            # Input section description
            case_description = self.report_content.case_description_dict.get(
                    case_key,
                    'No case description\n')

            chapters_content += \
                case_description_template.format(
                    case_description=case_description)

            # Input figure sections
            for section in case_figs:
                section_content = \
                    self.__generate_python_section(
                        sectionname=section,
                        case_figures=case_figs[section])

                # Add new content to existent one.
                # case_chapter = case_chapter + '\n' + section_content
                chapters_content += section_content

        if not chapters_content:
            error_mssg = '' + \
                'No LaTeX content was generated from ' + \
                '{}.'.format(cases_dict)
            raise ValueError(error_mssg)

        return chapters_content

    def __generate_python_section(self, sectionname: str, case_figures: list):
        """Generates a LaTeX string represeting a section with figures.

        Arguments:
            case_figures {list} -- List of figures (dictionaries)

        Returns:
            {str} -- LaTeX string representing the section and figures.
        """
        fig_sect_content = \
            '\\section{{{new_section_key}}}\n'.format(
                new_section_key=sectionname)
        if not case_figures:
            return fig_sect_content
        temp_path_key = 'FIG_PATH'
        temp_capt_key = 'FIG_CAPTION'
        temp_label_key = 'FIG_LABEL'
        figure_key = '{' + 'figure' + '}'
        section_template = '' + \
            '\t\\begin{}[!h]\n'.format(figure_key) + \
            '\t\t\\centering\n' + \
            '\t\t\\includegraphics[width=0.95' + \
            '\\textwidth]{{{}}}\n'.format(temp_path_key) + \
            '\t\t\\caption{{\\small {}}}\n'.format(temp_capt_key) + \
            '\t\t\\label{{fig:{}}}\n'.format(temp_label_key) + \
            '\t\\end{}\n'.format(figure_key)

        for figure in case_figures:
            # Get key values from dictionary.
            fig_path = figure.get(ReportHelper._fig_rel_path_key)
            fig_capt = figure.get(ReportHelper._fig_name_key)
            fig_key = figure.get(ReportHelper._fig_key_key)
            # Replace values in template.
            fig_sect = section_template.replace(temp_path_key, fig_path)
            fig_sect = fig_sect.replace(temp_capt_key, fig_capt)
            fig_sect = fig_sect.replace(temp_label_key, fig_key)
            # Add new content to existent.
            fig_sect_content = fig_sect_content + '\n' + fig_sect

        return fig_sect_content
