@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Note chaque clips\<nom>\veyra_out.mp4 et compare a la reference du clip.
set /p LABEL=Nom de la version testee (ex. 2.2.1-flow-v3) : 
py -3 veyra_quality.py suite clips --label "%LABEL%"
if errorlevel 1 (
  echo.
  echo *** REGRESSION : voir clips\summary_%LABEL%.md et les images worst_*.png ***
) else (
  echo.
  echo OK : pas de regression. Resume dans clips\summary_%LABEL%.md
)
pause
