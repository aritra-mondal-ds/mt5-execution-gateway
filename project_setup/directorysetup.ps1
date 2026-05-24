$base = "C:\Users\aritr\Personal Projects\TradingBotMT5"

$dirs = @(
    "config",
    "data\raw", "data\features", "data\signals",
    "feeds",
    "features",
    "strategies",
    "risk",
    "execution",
    "backtesting",
    "models\registry",
    "monitoring",
    "notebooks\research", "notebooks\backtest_reports",
    "tests"
)

foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path "$base\$d" | Out-Null
    New-Item -ItemType File -Force -Path "$base\$d\__init__.py" | Out-Null
}

# Root files
@("main.py","requirements.txt",".gitignore") | ForEach-Object {
    New-Item -ItemType File -Force -Path "$base\$_" | Out-Null
}