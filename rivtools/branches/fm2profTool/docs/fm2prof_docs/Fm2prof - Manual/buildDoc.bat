pdflatex "Fm2prof - Manual.tex"
bibtex "Fm2prof - Manual"
pdflatex "Fm2prof - Manual.tex"
pdflatex "Fm2prof - Manual.tex" > FM2PROF_MAN_Log.txt
xcopy FM2PROF_MAN_Log.txt "..\BuildLogs" /Y
xcopy *.pdf .. /Y
