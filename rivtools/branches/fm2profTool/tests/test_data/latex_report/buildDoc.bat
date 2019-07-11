pdflatex "acceptance_report.tex"
bibtex "acceptance_report"
pdflatex "acceptance_report.tex"
pdflatex "acceptance_report.tex" > a_r_Log.txt
xcopy a_r_Log.txt "..\BuildLogs" /Y
xcopy *.pdf .. /Y
