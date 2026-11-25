@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Installe ce qu'il faut pour les tests de qualite Veyra (une seule fois).
where py >nul 2>nul || winget install -e --id Python.Python.3.12
where ffmpeg >nul 2>nul || winget install -e --id Gyan.FFmpeg
py -3 -m pip install -r requirements.txt
echo.
echo Termine. Si ffmpeg vient d'etre installe, ferme et rouvre cette fenetre.
pause
