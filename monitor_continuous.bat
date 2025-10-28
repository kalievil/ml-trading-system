@echo off
echo 🤖 ML Trading Continuous Monitor
echo ================================
echo.
echo Starting continuous monitoring (1 minute intervals)...
echo Press Ctrl+C to stop
echo.
python -c "
from ml_trading_reporter import MLTradingReporter
reporter = MLTradingReporter()
reporter.run_continuous_monitoring(1)
"
