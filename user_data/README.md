
# Optimized XGBoost Freqtrade System

This system integrates the OptimizedHighReturnSystem algorithm with Freqtrade's architecture.

## Files Structure

- `strategies/OptimizedXGBoostStrategy.py` - Main strategy implementation
- `config_optimized_testnet.json` - Binance Testnet configuration
- `config_dry_run.json` - Dry-run configuration for backtesting
- `models/` - Trained XGBoost models and scaler
- `data/binance/` - Market data in Freqtrade format
- `custom_backtest.py` - Custom backtesting wrapper

## Usage

### Backtesting
```bash
freqtrade backtesting --config user_data/config_dry_run.json --strategy OptimizedXGBoostStrategy
```

### Paper Trading (Binance Testnet)
```bash
freqtrade trade --config user_data/config_optimized_testnet.json --strategy OptimizedXGBoostStrategy
```

### Custom Backtesting
```bash
python user_data/custom_backtest.py
```

### Web UI
```bash
freqtrade webserver --config user_data/config_optimized_testnet.json
```
Then open http://127.0.0.1:8080

## Configuration

Before running with Binance Testnet:
1. Get API keys from https://testnet.binance.vision/
2. Update `config_optimized_testnet.json` with your API keys
3. Ensure you have testnet USDT balance

## Model Training

The system automatically trains XGBoost models on first run. Models are saved in `user_data/models/` and reused for subsequent runs.

## Performance

The optimized system targets:
- 60-80% annual returns
- 90%+ win rate
- 1.8% stop loss, 1.5% take profit
- Dynamic leverage up to 5x
- Position sizing 30-35% of capital
