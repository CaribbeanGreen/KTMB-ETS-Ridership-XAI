@echo off
rem Opens the KTMB ETS dashboard with the one-click update buttons. Keep this window open while you use it.
cd /d "%~dp0"
python serve.py
if errorlevel 1 pause
