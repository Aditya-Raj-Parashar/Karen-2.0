@echo off
echo __ Karen Enrollment _________________________________________________________
echo.

echo [0/3] Clearing existing gallery...
.\.venv\Scripts\python.exe main.py --clear-gallery
echo.

echo [1/3] Enrolling Lakshay (c1)...
.\.venv\Scripts\python.exe main.py --enroll people/lakshay.mp4 --label "Lakshay"
echo.

echo [2/3] Enrolling Manisha (c2)...
.\.venv\Scripts\python.exe main.py --enroll people/manisha.mp4 --label "Manisha"
echo.

echo [3/3] Enrolling Aditya (c3)...
.\.venv\Scripts\python.exe main.py --enroll people/aditya.mp4 --label "Aditya"
echo.

echo __ Done. Run run_cctv.bat to start tracking. ________________________________
pause
