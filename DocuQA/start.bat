@echo off
cd /d "%~dp0"
uv run --locked streamlit run app.py
if errorlevel 1 pause
