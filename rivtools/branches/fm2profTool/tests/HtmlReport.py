import os
import shutil
import tests.ReportHelper as ReportHelper


class HtmlReport:

    __html_report_name = 'index.html'

    __html_template_dir = 'html_report'
    __html_idx_temp_name = 'index_template.html'
    __html_sec_temp_name = 'section_template.html'
    __html_fig_temp_name = 'figure_template.html'
    __html_car_temp_name = 'carousel_sec_template.html'

    __sec_name_key = 'SECTION_NAME'
    __sec_link_key = 'SECTION_LINK_LINK'
    __sec_link_text_key = 'SECTION_LINK_TEXT'
    __sec_subsections_key = 'SUB_SECTIONS_CONTENT'

    __sec_car_id_key = 'SECTION_CAROUSEL_ID'
    __sec_car_ind_key = 'SECTION_CAROUSEL_INDICATORS'
    __sec_car_img_key = 'SECTION_CAROUSEL_IMGS'

    def _generate_html_report(self, target_dir: str, scenarios: dict):
        """Generates an HTML report based on the contents of the dictionary.

        Arguments:
            target_dir {str} -- Target directory where to generate report.
            scenarios {dict} -- Dictionary with values for the report.

        Returns:
            {str} -- HTML file path.
        """
        # Generate html code from dictionary based on templates.
        html_reports = self.__get_html_index_content(scenarios)

        # Clean report directory in case it already exists.
        report_dir = os.path.join(target_dir, self.__html_template_dir)
        if os.path.exists(report_dir):
            shutil.rmtree(report_dir)
        os.makedirs(report_dir)

        # Save new content to html file path.
        for html_report in html_reports:
            html_path = os.path.join(report_dir, html_report)
            html_content = html_reports[html_report]
            with open(html_path, 'w') as f:
                f.write(html_content)
            # Verify HTML has been generated.
            if not os.path.exists(html_path):
                raise Exception(
                    'HTML report was not generated at {}.'.format(html_path))

        return html_reports.keys()

    def __get_html_template(self, template_name: str):
        """Gets the HTML template content from their folder.

        Arguments:
            templates_dir {str} -- Directory containing the templates.

        Returns:
            {tuple} -- Tuple of three templates.
        """
        # Get the HTML Templates folder.
        template_dir = ReportHelper._get_template_folder(
            self.__html_template_dir)
        if not os.path.exists(template_dir):
            raise IOError('Template directory could not be found at {}'.format(
                self.__template_dir))

        template_path = os.path.join(template_dir, template_name)
        template_content = self.__get_html_template_content(template_path)

        return template_content

    def __get_html_template_content(self, file_path: str):
        """Gets the content of a template as a string.

        Arguments:
            file_path {str} -- Location of the template.

        Raises:
            IOError: When file path is not valid.

        Returns:
            {str} -- String with HTML code.
        """
        if not os.path.exists(file_path):
            raise IOError('HTML Template not found at {}'.format(file_path))

        template_content = ''
        with open(file_path) as f:
            template_content = f.read()

        return template_content

    def __get_html_index_content(self, cases_dict: dict):
        """Generates HTML code with all content of the dictionary given.

        Arguments:
            cases_dict {dict} -- Dictionary with the report content

        Returns:
            {tuple} -- Tuple containing HTML code with all sections
                        and a list with one element per HTML section.
        """
        # Get cases HTML formatted content.
        sections_html, sections_pages = self.__get_html_sections(cases_dict)

        # Insert generated HTML into highest template.
        reports = {}
        idx_html = self.__get_html_template(self.__html_idx_temp_name)
        report_html = idx_html.replace('PYTHON_CODE_HERE', sections_html)
        reports[self.__html_report_name] = report_html
        # Create an entry per each section page.
        for page in sections_pages:
            page_html = sections_pages[page]
            content_html = idx_html.replace('PYTHON_CODE_HERE', page_html)
            reports[page] = content_html

        return reports

    def __get_html_sections(self, cases_dict: dict):
        """Generates all HTML sections based on the content of the dictionary.

        Arguments:
            cases_dict {dict} -- Dictionary of report content.

        Returns:
            {tuple} -- Tuple containing HTML code with all sections
                        and a dictionary with one element per HTML section.
        """
        html_sections = ''
        html_page_sections = {}
        if not cases_dict:
            return html_sections, html_page_sections

        sec_temp = self.__get_html_template(self.__html_sec_temp_name)
        car_temp = self.__get_html_template(self.__html_car_temp_name)
        for case in cases_dict:
            case_dict = cases_dict[case]
            # Get dictionaries of mapped template-content.
            carousel_dict = self.__get_carousel_map_values(case_dict)
            section_dict = self.__get_sec_map_values(case_dict)

            # Get HTML content based on the templates.
            carousel_html = self.__set_content_in_template(
                car_temp, carousel_dict)
            section_html = self.__set_content_in_template(
                sec_temp, section_dict)

            # Set values to lists
            html_link = carousel_dict.get(self.__sec_link_key)
            html_page_sections[html_link] = section_html
            # Add new content to existent one.
            html_sections = html_sections + '\n' + carousel_html

        return html_sections, html_page_sections

    def __set_content_in_template(self, template: str, content: dict):
        """Sets all the content in the given template.

        Arguments:
            template {str} -- HTML string template.
            content {dict} -- String containing a key-value map of templates.

        Returns:
            {str} -- HTML string with injected content.
        """
        if not template or not content:
            return

        for temp_key, case_value in content.items():
            template = template.replace(temp_key, case_value)

        return template

    def __get_carousel_map_values(self, case_dict: dict):
        """Generates a dictionary of keys with values to replace in the templates.

        Arguments:
            case_dict {dict} -- Dictionary of values for the templates.

        Returns:
            {dic} -- Mapped dictionary of Template-Keys and Values
        """
        # Get values from dictionary.
        case_name = case_dict.get(ReportHelper._case_lab_key)
        case_figs = case_dict.get(ReportHelper._case_fig_key)
        case_key = case_dict.get(ReportHelper._case_key_key)

        # Get carousel elements.
        carousel_elements = self.__get_section_carousel(case_figs)
        car_ind, car_img = carousel_elements
        html_indicators = ''.join(car_ind)
        html_carousel_imgs = ''.join(car_img)

        # Get default dictionary
        combined_dict = self.__get_default_sec_values(case_dict)

        # Set new values to dictionary.
        combined_dict[self.__sec_car_id_key] = case_key
        combined_dict[self.__sec_car_ind_key] = html_indicators
        combined_dict[self.__sec_car_img_key] = html_carousel_imgs

        # Update values from default dictionary.
        html_link = case_name.replace(' ', '') + '.html'
        text_back = 'See {} results in detail.'.format(case_name)
        combined_dict[self.__sec_link_key] = html_link
        combined_dict[self.__sec_link_text_key] = text_back

        return combined_dict

    def __get_sec_map_values(self, case_dict: dict):
        """Generates a dictionary of keys with values to replace in the templates.

        Arguments:
            case_dict {dict} -- Dictionary of values for the templates.

        Returns:
            {dic} -- Mapped dictionary of Template-Keys and Values
        """
        # Get values from dictionary.
        case_figs = case_dict.get(ReportHelper._case_fig_key)

        # Get default dictionary.
        combined_dict = self.__get_default_sec_values(case_dict)

        # Get section elements
        html_figs = self.__get_html_figures(case_figs)

        # Update values from default dictionary.
        combined_dict[self.__sec_subsections_key] = html_figs

        return combined_dict

    def __get_default_sec_values(self, case_dict: dict):
        """Generates a dictionary of keys with values to replace in the templates.

        Arguments:
            case_dict {dict} -- Dictionary of values for the templates.

        Returns:
            {dic} -- Mapped dictionary of Template-Keys and Values
        """
        # Get values from dictionary.
        case_name = case_dict.get(ReportHelper._case_lab_key)
        case_figs = case_dict.get(ReportHelper._case_fig_key)
        case_key = case_dict.get(ReportHelper._case_key_key)

        # Set dictionary.
        text_back = 'Go back to overview.'
        default_dict = {
            self.__sec_name_key: case_name,
            self.__sec_link_key: self.__html_report_name,
            self.__sec_link_text_key: text_back,
        }

        return default_dict

    def __get_section_carousel(self, figures_list: list):
        indicators = []
        images = []
        for idx, figure in enumerate(figures_list):
            indicator = self.__get_carousel_indicator(idx)
            image = self.__get_carousel_img(figure, idx)
            if indicator and image:
                indicators.append(indicator)
                images.append(image)
        return indicators, images

    def __get_carousel_img(self, figure: dict, idx: int):
        if not figure:
            return ''
        # Get values from the figure dictionary.
        alt_text = figure.get(ReportHelper._fig_name_key)
        fig_path = figure.get(ReportHelper._fig_rel_path_key)
        fig_caption = figure.get(ReportHelper._fig_name_key)

        # Determine class of parent div.
        div_class = 'carousel-item'
        if idx == 0:
            div_class = 'carousel-item active'

        # Create HTML code for the figure
        element = '' + \
            '\t\t\t\t<div class="{}">\n'.format(div_class) + \
            '\t\t\t\t<img src="{}"'.format(fig_path) + \
            ' class="d-block w-100"' + \
            ' alt="{}">\n'.format(alt_text) +\
            '\t\t\t\t</div>\n'
        return element

    def __get_carousel_img_caption(self, caption: str):
        element = '' + \
            '<div class="carousel-caption d-none d-md-block">\n' + \
            '\t\t\t\t\t<h5>{}</h5>'.format(caption) + \
            '</div>\n'
        return element

    def __get_carousel_indicator(self, idx: int):
        li_class = ''
        if idx == 0:
            li_class = 'class="active" '
        indicator = '' + \
            '\t\t\t\t<li {}'.format(li_class) + \
            'data-target="carouselIndicators" ' + \
            'data-slide-to="{}"/>\n'.format(idx)

        return indicator

    def __get_html_figures(self, case_figures: list):
        """Generates HTML code of formatted figures using a template.

        Arguments:
            case_figures {list} -- List of figures (dictionaries)

        Returns:
            {str} -- HTML formatted code.
        """
        html_figures = ''
        if not case_figures:
            return html_figures
        fig_capt_key = 'FIGURE_NAME_HERE'
        fig_path_key = 'FIGURE_PATH'
        fig_alt_txt_key = 'FIGURE_ALT_TEXT'
        template = self.__get_html_template(self.__html_fig_temp_name)
        for figure in case_figures:
            # Get key values from dictionary.
            fig_path = figure.get(ReportHelper._fig_rel_path_key)
            fig_capt = figure.get(ReportHelper._fig_name_key)
            fig_key = figure.get(ReportHelper._fig_key_key)
            # Set values in template.
            fig_html = template.replace(fig_capt_key, fig_capt)
            fig_html = fig_html.replace(fig_path_key, fig_path)
            fig_html = fig_html.replace(fig_alt_txt_key, fig_capt)
            # Add to existent content.
            html_figures = html_figures + '\n' + fig_html

        return html_figures
