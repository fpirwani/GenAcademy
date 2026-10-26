@echo off
cd /d "%~dp0"
uv run streamlit run app.py --server.port 8502
