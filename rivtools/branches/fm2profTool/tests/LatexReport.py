import os
import shutil
import tests.ReportHelper as ReportHelper


class LatexReport:
    __latex_template_dir = 'latex_report'
    __latex_template_name = 'report_template.tex'
    __latex_report_name = 'acceptance_report'

    def __init__(self):
        pass

    def _convert_to_pdf(self, report_path: str):
        """Given a path to the LaTeX file, its PDF is generated
        through system calls

        Arguments:
            report_path {str} -- Path to LaTeX file
        """
        report_dir = os.path.dirname(report_path)
        current_wc = os.getcwd()
        os.chdir('{}'.format(report_dir))
        try:
            pycall = 'pdflatex {}'.format(report_path)
            os.system('{} > ar_first_Log.txt'.format(pycall))
            os.system('{} > ar_first_toc_Log.txt'.format(pycall))
            os.system('{} > ar_last_Log.txt'.format(pycall))
            # os.system('buildDoc.bat')
        except Exception as e_info:
            print('Error while generating pdf: {}'.format(e_info))
        os.chdir('{}'.format(current_wc))

    def _generate_python_report(self, target_dir: str, report_dict: dict):
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
        template_dir = ReportHelper._get_template_folder(
            self.__latex_template_dir)
        if not os.path.exists(template_dir):
            raise IOError('\
                Template directory could not be found at {}\
                    '.format(template_dir))

        # Clean the report dir (target)
        report_dir = os.path.join(target_dir, self.__latex_template_dir)
        if os.path.exists(report_dir):
            shutil.rmtree(report_dir)

        # Copy new templates.
        shutil.copytree(template_dir, report_dir, False, None)
        template_path = os.path.join(report_dir, self.__latex_template_name)
        assert os.path.exists(template_path), '\
            Latex template could not be found on path \
            {}.'.format(template_path)

        # Generate LaTeX report file.
        latex_report = self.__latex_report_name + '.tex'
        latex_report_path = os.path.join(report_dir, latex_report)
        shutil.move(template_path, latex_report_path)
        assert os.path.exists(latex_report_path), '\
            Latex report could not be found on path \
            {}.'.format(latex_report_path)

        # Create LaTeX content from dictionaries.
        latex_content = self.__generate_python_chapters(report_dict)
        assert latex_content, '\
            No LaTeX content was generated from {}'.format(report_dict)

        # Read template content.
        with open(latex_report_path) as f:
            file_str = f.read()

        # Replace template content.
        latex_content_key = 'PYTHON_CODE_HERE'
        report_dict = file_str.replace(latex_content_key, latex_content)

        # Add new content to LaTeX report.
        with open(latex_report_path, 'w') as f:
            f.write(report_dict)

        assert os.path.exists(latex_report_path), '\
            LaTeX report not found at {}'.format(latex_report_path)

        return latex_report_path

    def __generate_python_chapters(self, cases_dict: dict):
        """Generates a LaTeX string containing a chapter per case.

        Arguments:
            cases_dict {dict} -- Dictionary of cases and their result figures.

        Returns:
            {str} -- LaTeX string containing chapter and figures.
        """
        temp_name_key = 'CASE_NAME'
        temp_key_key = 'CASE_KEY'
        template = '' + \
            '\\chapter{' + temp_name_key + '}\n' + \
            '\\label{sec:' + temp_key_key + '}'
        chapter_content = ''
        for case in cases_dict:
            case_dict = cases_dict[case]
            # Get values from dictionary.
            case_name = case_dict.get(ReportHelper._case_lab_key)
            case_key = case_dict.get(ReportHelper._case_key_key)
            case_figs = case_dict.get(ReportHelper._case_fig_key)
            # Replace values in template.
            case_chapter = template.replace(temp_name_key, case_name)
            case_chapter = case_chapter.replace(temp_key_key, case_key)
            section_content = self.__generate_python_section(case_figs)
            # Add new content to existent one.
            case_chapter = case_chapter + '\n' + section_content
            chapter_content = chapter_content + '\n' + case_chapter

        return chapter_content

    def __generate_python_section(self, case_figures: list):
        """Generates a LaTeX string represeting a section with figures.

        Arguments:
            case_figures {list} -- List of figures (dictionaries)

        Returns:
            {str} -- LaTeX string representing the section and figures.
        """
        fig_sect_content = ''
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
