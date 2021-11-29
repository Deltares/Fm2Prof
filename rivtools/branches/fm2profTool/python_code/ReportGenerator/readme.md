# How to use the report generator:
- Define an inherited class of ReportContent on your own solution. You will be forced to defined the abstract properties (please do not redefined the ReportContent class):
    - Project version.
    - Project name.
    - Project number.
    - Authors
    - Cases and their descriptions.
    - Scenarios names (can be added through the constructor keyword scenarios_ids)
    - Location of the test data (can be added through the constructor keyword data_dir)

- The output data is expected such as:
    /data_dir/scenario/Figures/figure_type

# To create a LaTeX / HTML reports:
    my_report_content = \
        my_custom_ReportContent(
            scenarios_ids=[List of scenarios],
            data_dir=/Path/To/My/OutputCases)
    latex_report = LatexReport(my_report_content)
    latex_report.generate_latex_report(target_directory)

    html_report = HtmlReport(my_report_content)
    html_report.generate_html_report(target_directory)
    
# To generate a LaTeX PDF:
    - To convert to PDF simply call the static method:
    LatexReport.convert_to_pdf(dir_path) with the directory path of your latex report.