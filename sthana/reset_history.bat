@echo off
REM ============================================================
REM  Sthana - wipe all saved history and start fresh
REM  Deletes outputs\ (runs, crops, CSVs). Optionally uploads\.
REM  Put this file in the seat_monitor folder and double-click it.
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo  Sthana - reset history
echo  ----------------------
echo  Folder: %CD%
echo.

REM --- make sure app.py is not running (it listens on port 5000) ---
netstat -ano | findstr /r /c:":5000 .*LISTENING" >nul
if %errorlevel%==0 (
    echo  The app is still running on port 5000.
    echo  Stop it first: press Ctrl+C in the window running "python app.py".
    echo.
    pause
    exit /b 1
)

REM --- confirm ---
set /p ok= Delete ALL saved runs, crops and CSV files? (y/n):
if /i not "%ok%"=="y" (
    echo  Cancelled. Nothing was deleted.
    pause
    exit /b 0
)

if exist "outputs" (
    rmdir /s /q "outputs"
    if exist "outputs" (
        echo  Could not delete outputs\ - close any open images/CSVs from it and try again.
        pause
        exit /b 1
    )
    echo  Deleted outputs\
) else (
    echo  outputs\ was already empty.
)

set /p up= Also delete uploaded videos in uploads\? (y/n):
if /i "%up%"=="y" (
    if exist "uploads" rmdir /s /q "uploads"
    echo  Deleted uploads\
)

echo.
set /p run= Start the app now? (y/n):
if /i "%run%"=="y" (
    if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"
    python app.py
) else (
    echo  Done. Start it later with: python app.py
    pause
)
endlocal
