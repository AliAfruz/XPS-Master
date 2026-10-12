@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"
set "PYTHONPATH=%PROJECT_DIR%src;%PYTHONPATH%"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -c "import xps_afruz, PySide6, scipy" >nul 2>&1
    if not errorlevel 1 goto run_venv
)

py -3.12 -c "import xps_afruz, PySide6, scipy" >nul 2>&1
if not errorlevel 1 goto run_py312
py -3.11 -c "import xps_afruz, PySide6, scipy" >nul 2>&1
if not errorlevel 1 goto run_py311
py -3.10 -c "import xps_afruz, PySide6, scipy" >nul 2>&1
if not errorlevel 1 goto run_py310

rem A moved .venv can still contain usable dependencies for a compatible
rem bundled interpreter. Validate imports before selecting this last fallback.
set "BUNDLED_PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%BUNDLED_PYTHON_EXE%" goto no_environment
set "PYTHONPATH=%PROJECT_DIR%src;%PROJECT_DIR%.venv\Lib\site-packages;%PYTHONPATH%"
"%BUNDLED_PYTHON_EXE%" -c "from xps_afruz.ui.app import main; import tkinter, numpy; assert int(numpy.__version__.split('.')[0]) >= 2" >nul 2>&1
if not errorlevel 1 goto run_bundled

:no_environment
echo.
echo No working XPS-master Python environment was found.
echo The existing .venv may refer to a Python version that was removed.
echo Run this once from the project folder with an installed Python version:
echo   py -3.11 -m venv .venv
echo   .venv\Scripts\python.exe -m pip install -e ".[dev]"
echo.
pause
exit /b 1

:run_venv
"%PYTHON_EXE%" "%PROJECT_DIR%run.py"
goto finished

:run_py312
py -3.12 "%PROJECT_DIR%run.py"
goto finished

:run_py311
py -3.11 "%PROJECT_DIR%run.py"
goto finished

:run_py310
py -3.10 "%PROJECT_DIR%run.py"
goto finished

:run_bundled
"%BUNDLED_PYTHON_EXE%" "%PROJECT_DIR%run.py"

:finished
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
    echo.
    echo XPS-master closed successfully.
) else (
    echo.
    echo XPS-master stopped with exit code %EXIT_CODE%.
)

echo.
pause
exit /b %EXIT_CODE%
