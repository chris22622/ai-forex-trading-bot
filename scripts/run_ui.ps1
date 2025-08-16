# PowerShell script to run the AI Forex Trading Bot UI
Write-Host "🚀 Starting AI Forex Trading Bot UI..." -ForegroundColor Green
Write-Host "Installing dependencies..." -ForegroundColor Yellow
& python -m pip install -r requirements.txt

Write-Host "Launching Streamlit dashboard..." -ForegroundColor Yellow
Write-Host "🌐 The web UI will open in your browser at http://localhost:8501" -ForegroundColor Cyan
& python -m streamlit run ui/streamlit_app.py
