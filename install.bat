@echo off
setlocal enabledelayedexpansion

:: Check if venv exists and is activated
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    
    :: Check if required packages are installed
    pip freeze > temp_requirements.txt
    findstr /C:"torch" temp_requirements.txt > nul
    if !errorlevel! equ 0 (
        echo Virtual environment is set up and PyTorch is installed.
        echo Launching venv console...
        del temp_requirements.txt
        goto launch_console
    ) else (
        echo PyTorch is not installed in the virtual environment.
        goto install
    )
) else (
    echo Virtual environment not found.
    goto install
)

:install
echo Starting installation process...

echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing requirements...
pip install -r requirements.txt

:launch_console
:: Create a temporary batch file
(
echo @echo off
echo call venv\Scripts\activate.bat
echo nvcc --version
echo.
echo echo Venv activated!
echo.
echo echo Please Install the correct version of PyTorch for your system from:
echo echo https://pytorch.org/get-started/locally/
echo.
echo echo You can now enter the PyTorch installation command.
echo cmd /k
) > temp_launch.bat

:: Start the new window using the temporary batch file
start cmd /c temp_launch.bat

if "%1" neq "install" (
    echo Cloning repository...
    git clone https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two temp_repo

    echo Copying files...
    xcopy /E /I /Y temp_repo\cgrkzexw-599808 cgrkzexw-599808

    echo Cleaning up...
    rmdir /S /Q temp_repo

    echo Installation complete!
)

:end
echo Deleting temporary files...
if exist temp_launch.bat del temp_launch.bat
pause