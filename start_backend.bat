@echo off
echo Starting Al-Fatih Alal-Imaam Backend AI Engine...
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found at .venv\Scripts\python.exe!
    echo Please make sure you have run the setup script.
    pause
    exit /b
)

.venv\Scripts\python.exe backend.py

echo.
echo [ERROR] Backend crashed or closed!
pause
