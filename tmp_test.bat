@echo off
.\.venv\Scripts\python.exe main.py --clear-gallery
.\.venv\Scripts\python.exe main.py --enroll people/lakshay.mp4 --label "Lakshay"
.\.venv\Scripts\python.exe main.py --enroll people/manisha.mp4 --label "Manisha"
.\.venv\Scripts\python.exe main.py --enroll people/aditya.mp4 --label "Aditya"
