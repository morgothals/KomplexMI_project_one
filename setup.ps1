# ------------------------------
# setup_project.ps1
# Creates project folder structure for the crypto AI project
# ------------------------------

# Root folder (you can rename)
$root = "crypto_ai_project"
New-Item -ItemType Directory -Force -Path $root | Out-Null
Set-Location $root

# --- Data folder and files ---
New-Item -ItemType Directory -Force -Path "data" | Out-Null
@("market_data.csv", "news_data.csv", "sentiment_data.csv") | ForEach-Object {
    New-Item -ItemType File -Force -Path ("data\" + $_) | Out-Null
}

# --- Models folder and files ---
New-Item -ItemType Directory -Force -Path "models" | Out-Null
@("sentiment_model.pkl", "forecast_model.h5") | ForEach-Object {
    New-Item -ItemType File -Force -Path ("models\" + $_) | Out-Null
}

# --- Modules folder and files ---
New-Item -ItemType Directory -Force -Path "modules" | Out-Null
@("data_collector.py", "sentiment_analyzer.py", "forecast_model.py", "advisor.py") | ForEach-Object {
    New-Item -ItemType File -Force -Path ("modules\" + $_) | Out-Null
}

# --- App folder structure ---
New-Item -ItemType Directory -Force -Path "app\templates" | Out-Null
New-Item -ItemType Directory -Force -Path "app\static" | Out-Null
New-Item -ItemType File -Force -Path "app\dashboard.py" | Out-Null

# --- Root files ---
New-Item -ItemType File -Force -Path "README.md" | Out-Null
New-Item -ItemType File -Force -Path "main.py" | Out-Null

# --- project.json ---
$jsonContent = @{
    project_name = "Crypto AI Advisor"
    description  = "AI-based cryptocurrency investment advisor using sentiment analysis and forecasting models."
    author       = "Team Tajti"
    python_version = ">=3.10"
    dependencies = @(
        "pandas>=2.2",
        "numpy>=1.26",
        "tensorflow>=2.15",
        "scikit-learn>=1.5",
        "yfinance>=0.2",
        "requests>=2.32",
        "beautifulsoup4>=4.12",
        "flask>=3.0"
    )
    modules = @(
        "modules/data_collector.py",
        "modules/sentiment_analyzer.py",
        "modules/forecast_model.py",
        "modules/advisor.py",
        "app/dashboard.py",
        "main.py"
    )
} | ConvertTo-Json -Depth 4

$jsonContent | Out-File -Encoding utf8 "project.json"

Write-Host "`n✅ Project structure created successfully in folder '$root'!"
