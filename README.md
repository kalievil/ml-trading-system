# ML Trading System - Binance Testnet

A fully automated ML-powered trading system that runs 24/7 on Binance Testnet using XGBoost ensemble models and advanced risk management.

## 🚀 Features

- **🤖 Fully Automated Trading**: ML algorithm executes trades automatically based on signals
- **📊 Advanced ML Models**: XGBoost ensemble with KDE market profile analysis
- **🛡️ Risk Management**: Stop-loss, take-profit, drawdown limits, daily loss limits
- **📈 Real-time Monitoring**: HTML reports and API endpoints for monitoring
- **🔧 Easy Deployment**: Ready for Render.com deployment

## 🏗️ Architecture

- **Backend**: FastAPI Python server (`trading_api.py`)
- **ML Engine**: OptimizedHighReturnSystem with XGBoost models
- **Frontend**: React webapp (optional, can use HTML reports)
- **Monitoring**: JSON data storage + HTML report generation

## 📁 Project Structure

```
├── trading_api.py              # Main FastAPI server
├── complete_high_return_optimized.py  # ML trading algorithm
├── ml_trading_reporter.py      # HTML report generator
├── requirements.txt            # Python dependencies
├── user_data/
│   ├── models/                # ML models and scalers
│   └── strategies/            # Trading strategies
├── paper-pilot-live-main/     # React frontend (optional)
└── trading_reports/           # Generated HTML reports
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Binance API
```bash
python trading_api.py
# Visit http://localhost:8000/docs
# Use /api/configure-binance endpoint
```

### 3. Enable Auto Trading
```bash
curl -X POST http://localhost:8000/api/enable-auto-trading
```

### 4. Monitor Performance
```bash
python ml_trading_reporter.py
# Choose option 2 for continuous monitoring
```

## 🌐 API Endpoints

- `GET /api/health` - System health check
- `POST /api/configure-binance` - Configure Binance API keys
- `GET /api/ml-signal` - Get current ML trading signal
- `POST /api/enable-auto-trading` - Enable automatic trading
- `POST /api/disable-auto-trading` - Disable automatic trading
- `GET /api/account-info` - Get account information
- `GET /api/positions` - Get current positions
- `GET /api/trade-history` - Get trade history
- `GET /api/trading-performance` - Get performance metrics

## 📊 ML Algorithm Details

- **Models**: XGBoost ensemble (aggressive, balanced, performance)
- **Features**: 19 technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Signal Generation**: Confidence-based trading signals
- **Risk Management**: Dynamic position sizing and leverage
- **Safety Controls**: MAE monitoring, drawdown limits

## 🛡️ Risk Management

- **Stop Loss**: 1.8% per trade
- **Take Profit**: 1.5% per trade
- **Max Drawdown**: 15% portfolio limit
- **Daily Loss Limit**: 5% daily max loss
- **MAE Monitoring**: Exits if unrealized loss > 2x stop-loss

## 🚀 Render Deployment

1. **Fork this repository**
2. **Connect to Render.com**
3. **Set environment variables**:
   - `BINANCE_API_KEY`: Your Binance Testnet API key
   - `BINANCE_API_SECRET`: Your Binance Testnet API secret
4. **Deploy as Web Service**

## 📈 Monitoring

The system generates HTML reports every 5 minutes with:
- Account balance and P&L
- Current positions
- Trade history
- ML signals and confidence
- Performance metrics

## ⚠️ Disclaimer

This is for educational purposes only. Trading involves risk. Use only on Binance Testnet for testing.

## 📝 License

MIT License - Feel free to use and modify for your trading experiments.
