@echo off
echo.
echo ===================================
echo  AI Forex Trading Bot Demo Setup
echo ===================================
echo.
echo 1. Starting Streamlit app...
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
start "" streamlit run ui/streamlit_app.py --server.port 8505
timeout /t 5
echo.
echo 2. Opening browser...
start "" http://localhost:8505
echo.
echo ===================================
echo  DEMO RECORDING INSTRUCTIONS
echo ===================================
echo.
echo 1. Switch to DEMO MODE in sidebar
echo 2. Click START button
echo 3. Press Win+Alt+R to record (15-20 seconds)
echo 4. Show chart updating and logs
echo 5. Press Win+Alt+R again to stop recording
echo.
echo When done:
echo - Convert video to GIF at ezgif.com
echo - Save as docs/demo.gif
echo - Run: git add docs/demo.gif ^&^& git commit -m "Add demo GIF" ^&^& git push
echo.
pause
