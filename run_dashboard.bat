@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Virtual environment not found. Create it and install requirements first.
  echo py -3.11 -m venv .venv
  echo .venv\Scripts\activate
  echo pip install -r requirements.txt
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run dashboard\app.py
pause
