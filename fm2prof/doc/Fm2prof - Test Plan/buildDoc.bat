pdflatex "Fm2prof - Test Plan.tex"
bibtex "Fm2prof - Test Plan"
pdflatex "Fm2prof - Test Plan.tex"
pdflatex "Fm2prof - Test Plan.tex" > FM2PROF_TP_Log.txt
xcopy FM2PROF_TP_Log.txt "..\BuildLogs" /Y
xcopy *.pdf .. /Y
