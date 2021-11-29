mkdir "BuildLogs"

cd "Fm2prof - Manual"
call buildDoc.bat
cd ..

cd "Fm2prof - Test Plan"
call buildDoc.bat
cd ..

cd "Fm2prof - Test Report"
call buildDoc.bat
cd ..

python parseLogs.py
