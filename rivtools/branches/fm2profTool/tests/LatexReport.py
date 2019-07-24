import os
import shutil
import tests.ReportHelper as ReportHelper

case_description_dict = {
'Case01rectangleKey':
r"""
This case test a basic rectangular profile. 
It is the simplest rectangular case only with a main channel. 
The width of the channel is 150m and the longitudinal length is 3000m with a slope of 1:3000.
We test the following items:
\begin{enumerate}
\item Correct generation of the rectangular cross-section
\item The roughness curve in the main channel
\item The water volume as function of water level
\end{enumerate}

The following output is expected
\begin{enumerate}
\item The geometry is expected to be approximatly identical to the true rectangular shape. The following deviations are expected due to the 'slope effect':
    \begin{enumerate}
        \item The first (most upstream) cross-section is expected to be almost identical
        \item The last cross-section is expected to have a bias near the bed level, whereby the 1D bed level is higher than the analytical (2D) bed level
        \item In between, we expect a small error near the bed level, which will result a not perfectly rectangular 1D profile, but exhibiting truncated corners.
    \end{enumerate}
\item The roughness curve is expected to follow a Manning curve with known n
\item The volume graph is expected to follow 2D results
\end{enumerate}
""",
'Case02compoundKey':
r"""
It has a symmetric 50m wide main channel and 50m floodplains on the both sides of the main channel (total width of 150m). The depth of the main channel is uniformly 2m. The longitudinal length is 3000m with a slope of 1:3000.

We test the following items:
\begin{enumerate}
\item Correct generation of a compound cross-section
\item The roughness curve in the main channel and floodplain
\item The water volume as function of water level
\end{enumerate}

The following output is expected
\begin{enumerate}
\item The geometry is expected to be approximatly identical to the true shape. The following deviations are expected due to the 'slope effect':
    \begin{enumerate}
        \item The first (most upstream) cross-section is expected to be almost identical
        \item The last cross-section is expected to have a bias near the bed level, whereby the 1D bed level is higher than the analytical (2D) bed level
        \item In between, we expect a small error near the bed level, which will result a not perfectly rectangular 1D profile in the lowest stage, but exhibiting truncated corners.
    \end{enumerate}
\item Two roughness curves that follow Manning curves with known n
\item The volume graph is expected to follow 2D results
\end{enumerate}
""",
'Case03threestageKey':
r"""
It has a symmetric 50m wide main channel and two different heights floodplains on the both sides of the main channel. The inner part of the floodplain is 2m from the bottom of the main channel, and the outer floodplain is 0.5m higher than the inner floodplain. Each part of the floodplain has 25m in width (total floodplain width is 100m). The longitudinal length is 3000m with a slope of 1:3000.

We test the following items:
\begin{enumerate}
\item Correct generation of a compound (three-stage) cross-section
\item The roughness curve in the main channel and floodplain
\item The water volume as function of water level
\end{enumerate}

The following output is expected
\begin{enumerate}
\item The geometry is expected to be approximatly identical to the true shape. The following deviations are expected due to the 'slope effect':
    \begin{enumerate}
        \item The first (most upstream) cross-section is expected to be almost identical
        \item The last cross-section is expected to have a bias near the bed level, whereby the 1D bed level is higher than the analytical (2D) bed level
        \item In between, we expect a small error near the bed level, which will result a not perfectly rectangular 1D profile in the lowest stage, but exhibiting truncated corners.
    \end{enumerate}
\item The roughness curve for the main channel is expected to follow a Manning curve with known n. 
The second (floodplain) curve is expected to be a compound curve, build up from known curves from the two floodplain stages. 
\item The volume graph is expected to follow 2D results
\end{enumerate}
""",
'Case04storageKey':
r"""
The basic geometry is the same as Case 02. 
In addition, infinitely high walls (a.k.a. thin dams in FM) are placed at 1250 and 1750m from the inlet perpendicular to the flow direction on one side of the floodplain. 
The area between two walls is considered a storage area because the flow is significantly slower than others.

We test the following items:
\begin{enumerate}
\item Correct generation of a compound (two-stage) cross-section
\item The roughness curve in the main channel and floodplain
\item The water volume as function of water level
\item At cross-section 1500, correct estimation of the width of the storage
\end{enumerate}

The following output is expected
\begin{enumerate}
\item For volume, rougness and geometry, this case follows case 2, with the exception of:
\item At cross-section 1500, the flow width should be smaller than the total width. 
\end{enumerate}
""",
'Case05dykeKey':
r"""
The basic geometry is the same as Case 02, but there is compartimentalistation due the embankments ('summer dikes'). 
At the boundary of the main channel and the floodplain, there are 1m high embankments on the both side of the main channel.  
Therefore, the water flows into the floodplain after the water depth reaches 3m instead of 2m in Case 02.

Expected outcome:
We test the following items:
\begin{enumerate}
\item Correct generation of a compound (two-stage) compartimentalised cross-section.
\item The water volume as function of water level
\end{enumerate}

The following output is expected
\begin{enumerate}
\item The generated cross-section should be comparable to case 2, but with the floodplain level at the higher of the embankment. 
\item The volume graphs should show that the applied volume correction is able to correctly follow 2D results. 
\end{enumerate}

\Note{The roughness values are shown for this case, but it is not (yet) known which values should be expected in this case.}

""",
'Case06plassenKey':
r"""
The basic geometry is the same as Case 02. 
A deep lake is located on the floodplain between 1250m and 1750m of the domain. 
The width of the lake is 25m, and it takes up the outer half of one side of the floodplain. 
Although the depth of the lake is approximately 10m from the floodplain, it does not influence the cross-section but roughness.

Expected outcome:
We test the following items:
\begin{enumerate}
\item Correct generation of a compound (two-stage) compartimentalised cross-section.
\item The roughness curve in the main channel and floodplain
\end{enumerate}

The following output is expected
\begin{enumerate}
\item The generated cross-section should be comparable to case 2. The lake should not affect geometry.
\item At cross-section 1500, the roughness should be high at water level under the floodplain level, then sharply decrease as the water level exceeds the floodplain level. 
\end{enumerate}


""",
'Case07triangularKey':
r"""
It has a triangular 2D grid in FM instead of rectangular 2D grid like the rest of the cases, and the overall geometry is similar to Case 02. 
The total width of the domain is 500m and the length is 10000m. 
The most part of the main channel has the width of 200m, and the floodplain has 150m at each side; however, the main channel width is increased to 250m near the inlet and outlet of the domain due to the geometrical limitation to represent the rectangular domain with “almost” regular triangles. 
It has a slope of 1:5000.

We test the following items:
\begin{enumerate}
\item Correct generation of the rectangular cross-section
\item The roughness curve in the main channel
\item The water volume as function of water level
\end{enumerate}

The following output is expected
\begin{enumerate}
\item The same expectation as case 2. However, inaccuracies are expected due to the inefficient triangular grid given rectangular geometry. 
\end{enumerate}
""",
'Case08waalKey':
r"""
The Waal case is a 'real world' study case of the River Waal. 
In this case, the 'true' geometry and roughness are not known. 
In this test, we directly compare 1D model results with 2D model results. 

We test the following items:
\begin{enumerate}
\item Whether output generated by FM2PROF can be succesfully used as input for the 1D model
\item The results of the 1D model compared to results of the 2D model at selected locations. 
\end{enumerate}

The following output is expected
\begin{enumerate}
\item A graph showing the overall statistics of deviations between 1D and 2D model results
\item Graphs of water level over time, showing 1D and 2D model results, as well as the deviation between them. 
\end{enumerate}

"""
}









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
            pycall = 'pdflatex {}/{}.tex'.format(os.path.join(current_wc, report_dir), self._LatexReport__latex_report_name)
            os.system('{} > ar_first_Log.txt'.format(pycall))
            os.system('{} > ar_first_toc_Log.txt'.format(pycall))
            os.system('{} > ar_last_Log.txt'.format(pycall))
            # os.system('buildDoc.bat')
        except Exception as e_info:
            print('Error while generating pdf: {}'.format(e_info))
        os.chdir('{}'.format(current_wc))
        pdf_name = self.__latex_report_name + '.pdf'
        pdf_path = os.path.join(report_dir, pdf_name)
        if not os.path.exists(pdf_path):
            raise Exception('PDF File was not generated.')

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
        with open(latex_report_path, encoding='UTF-8') as f:
            file_str = f.read()

        # Replace template content.
        latex_content_key = 'PYTHON_CODE_HERE'
        report_dict = file_str.replace(latex_content_key, latex_content)

        # Add new content to LaTeX report.
        with open(latex_report_path, 'w', encoding='UTF-8') as f:
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
        chapter_template = "\\chapter{{{temp_name_key}}}\n\\label{{{temp_key_key}}}\n"
        case_description_template = "\\section{{Case description}}\n{case_description}\n"
        chapter_content = ''
        for case in sorted(cases_dict):
            case_dict = cases_dict[case]
            # Get values from dictionary.
            case_name = case_dict.get(ReportHelper._case_lab_key)
            case_key = case_dict.get(ReportHelper._case_key_key)
            case_figs = case_dict.get(ReportHelper._case_fig_key)
            
            # Input Chapter header
            chapter_content += chapter_template.format(temp_name_key=case_name, temp_key_key=case_key)

            # Input section description
            case_description = case_description_dict.get(case_key)
            if case_description is not None:
                chapter_content += case_description_template.format(case_description=case_description)
            else:
                chapter_content += case_description_template.format(case_description="No case description\n")
            # Input figure sections
            for section in case_figs:
                section_content = self.__generate_python_section(section, case_figs[section])
            
                # Add new content to existent one.
                #case_chapter = case_chapter + '\n' + section_content
                chapter_content += section_content

        return chapter_content

    def __generate_python_section(self, sectionname: str, case_figures: list):
        """Generates a LaTeX string represeting a section with figures.

        Arguments:
            case_figures {list} -- List of figures (dictionaries)

        Returns:
            {str} -- LaTeX string representing the section and figures.
        """
        fig_sect_content = '\\section{{{new_section_key}}}\n'.format(new_section_key=sectionname)
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
