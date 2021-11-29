pdflatex "Fm2prof - Test Report.tex"
bibtex "Fm2prof - Test Report"
pdflatex "Fm2prof - Test Report.tex"
pdflatex "Fm2prof - Test Report.tex" > FM2PROF_TR_Log.txt
xcopy FM2PROF_TR_Log.txt "..\BuildLogs" /Y
xcopy *.pdf .. /Y
