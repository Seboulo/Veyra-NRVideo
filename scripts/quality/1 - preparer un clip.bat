@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Glisse un enregistrement Elgato (OBS) sur ce fichier.
if "%~1"=="" (
  echo Glisse un enregistrement video sur ce fichier.
  pause
  exit /b 1
)
set /p NAME=Nom du clip (ex. tete-rotation, hud, feuillage, nuit) : 
set /p START=Secondes a sauter au debut (Entree = 2) : 
if "%START%"=="" set START=2
py -3 veyra_quality.py prepare "%~1" --name "%NAME%" --out clips --start %START% --duration 12 --tag "%NAME%"
echo.
echo Exporte maintenant clips\%NAME%\input.mp4 avec Veyra en FSR 2X vers clips\%NAME%\veyra_out.mp4
pause
