@echo off
cd /d "%~dp0.."
call venv\Scripts\activate
cd EduSentiment-AI
python -m streamlit run app.py
pause

