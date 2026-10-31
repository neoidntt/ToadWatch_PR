@echo off
rem Double-click me on Windows: installs every library ToadWatch_PR needs.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3 is not installed. Get it from python.org and tick "Add to PATH". & pause & exit /b 1)
python -m venv .venv || (echo Could not create the virtual environment. & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || (echo Installation failed. & pause & exit /b 1)
python -m ipykernel install --user --name toadwatch --display-name "ToadWatch (.venv)"
echo.
echo Done. In VS Code open this folder, pick the kernel "ToadWatch (.venv)" and run notebooks\00_START_HERE.ipynb.
echo (PDF output needs LibreOffice: libreoffice.org)
pause
