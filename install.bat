@echo off
setlocal enabledelayedexpansion

:: Create and activate a virtual environment
python -m venv venv
call venv\Scripts\activate.bat

:: Install requirements
pip install -r requirements.txt

echo Installation of dependencies complete.
echo Please install PyTorch manually from https://pytorch.org/get-started/locally/
echo Select the appropriate CUDA version for your system.
pause