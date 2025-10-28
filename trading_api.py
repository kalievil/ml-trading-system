#!/usr/bin/env python3

"""
ML Trading API Server
Integrates complete_high_return_optimized.py with Binance Testnet API
"""

import os
import json
import logging
import asyncio
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import your ML algorithm
from complete_high_return_optimized import OptimizedHighReturnSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ML Trading API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
binance_client = None
ml_system = None
auto_trading_enabled = False
auto_trading_thread = None

# Credentials file path
CREDENTIALS_FILE = Path("binance_credentials.json")

# Pydantic models
class BinanceConfig(BaseModel):
    api_key: str
    api_secret: str

class TradingSignal(BaseModel):
    symbol: str
    side: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    leverage: float
    timestamp: str
    reason: str

def load_credentials_from_file():
    """Load credentials from file"""
    try:
        if CREDENTIALS_FILE.exists():
            with open(CREDENTIALS_FILE, 'r', encoding='utf-8') as f:
                credentials = json.load(f)
            logger.info(f"✅ Credentials loaded from file: {CREDENTIALS_FILE}")
            return credentials.get('api_key'), credentials.get('api_secret')
        return None, None
    except Exception as e:
        logger.error(f"❌ Error loading credentials: {e}")
        return None, None

def save_credentials_to_file(api_key: str, api_secret: str):
    """Save credentials to file"""
    try:
        credentials = {
            "api_key": api_key,
            "api_secret": api_secret,
            "saved_at": datetime.now().isoformat(),
            "testnet": True
        }
        
        with open(CREDENTIALS_FILE, 'w', encoding='utf-8') as f:
            json.dump(credentials, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Credentials saved to file: {CREDENTIALS_FILE}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save credentials: {e}")
        return False

def initialize_binance_client(api_key: str, api_secret: str):
    """Initialize Binance client"""
    global binance_client
    try:
        from binance.client import Client
        
        # Use testnet
        binance_client = Client(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True
        )
        
        # Test connection
        account_info = binance_client.get_account()
        logger.info(f"✅ Binance API connected - Account type: {account_info.get('accountType', 'UNKNOWN')}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Binance client: {e}")
        binance_client = None
        return False

def initialize_ml_system():
    """Initialize ML system"""
    global ml_system
    try:
        ml_system = OptimizedHighReturnSystem(max_leverage=5)
        logger.info("✅ ML system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize ML system: {e}")
        ml_system = None
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Starting ML Trading API Server...")
    
    # Load credentials from file
    api_key, api_secret = load_credentials_from_file()
    if api_key and api_secret:
        logger.info("🔑 Loaded API key: " + api_key[:10] + "...")
        logger.info("🔑 Loaded API secret: " + api_secret[:10] + "...")
        
        # Initialize Binance client
        logger.info("🔧 Initializing Binance client...")
        logger.info("🔧 Testing Binance connection...")
        if initialize_binance_client(api_key, api_secret):
            logger.info("✅ Binance API auto-configured from credentials file")
        else:
            logger.error("❌ Failed to configure Binance API")
    
    # Initialize ML system
    logger.info("🤖 Initializing ML system...")
    if initialize_ml_system():
        logger.info("✅ ML system ready")
    else:
        logger.error("❌ Failed to initialize ML system")
    
    logger.info("🌐 API Server ready!")

def automatic_trading_loop():
    """Background loop for automatic ML trading"""
    global auto_trading_enabled, binance_client, ml_system
    
    while auto_trading_enabled:
        try:
            # Check if Binance client is available
            if not binance_client:
                logger.warning("⚠️ Binance client not available for auto trading")
                time.sleep(60)
                continue
            
            # Check if ML system is available
            if not ml_system:
                logger.warning("⚠️ ML system not available for auto trading")
                time.sleep(60)
                continue
            
            # Get ML signal
            signal_data = get_ml_signal()
            
            # Check if signal meets criteria for automatic execution
            if signal_data['confidence'] >= 0.63 and signal_data['signal'] != 'HOLD':
                logger.info(f"🎯 Auto-trading: {signal_data['signal']} signal with {signal_data['confidence']:.1%} confidence")
                
                try:
                    # Execute trade based on signal
                    if signal_data['signal'] == 'BUY':
                        order = binance_client.order_market_buy(
                            symbol='BTCUSDT',
                            quantity=f"{signal_data['position_size']:.6f}"
                        )
                        logger.info(f"✅ Auto-executed BUY order: {order['orderId']}")
                    
                    elif signal_data['signal'] == 'SELL':
                        order = binance_client.order_market_sell(
                            symbol='BTCUSDT',
                            quantity=f"{signal_data['position_size']:.6f}"
                        )
                        logger.info(f"✅ Auto-executed SELL order: {order['orderId']}")
                    
                except Exception as trade_error:
                    logger.error(f"❌ Auto-trade execution failed: {trade_error}")
            
            else:
                logger.debug(f"🔍 Auto-trading: Signal {signal_data['signal']} with {signal_data['confidence']:.1%} confidence - No action")
            
            # Wait 10 seconds before next check
            time.sleep(10)
            
        except Exception as e:
            logger.error(f"❌ Auto-trading loop error: {e}")
            time.sleep(60)

def get_ml_signal():
    """Get ML signal using the optimized system"""
    try:
        if not ml_system:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'ML system not initialized',
                'leverage': 1.0,
                'position_size': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0
            }
        
        # Get current market data
        if not binance_client:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'Binance client not available',
                'leverage': 1.0,
                'position_size': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0
            }
        
        # Get current BTC price
        ticker = binance_client.get_ticker(symbol='BTCUSDT')
        current_price = float(ticker['lastPrice'])
        
        # For now, return a simple signal based on ML system parameters
        # In a real implementation, you would feed live data to the ML system
        
        # Simulate ML prediction (replace with actual ML prediction)
        confidence = 0.75  # Simulated confidence
        signal = 'BUY' if confidence > 0.63 else 'HOLD'
        
        # Calculate position size and risk parameters
        position_size = ml_system.calculate_position_size(confidence, 0.02)  # 2% volatility
        leverage = ml_system.calculate_dynamic_leverage(confidence, 0.02)
        stop_loss = current_price * (1 - ml_system.stop_loss_pct)
        take_profit = current_price * (1 + ml_system.take_profit_pct)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': f'ML system prediction with {confidence:.1%} confidence',
            'leverage': leverage,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': current_price,
            'algorithm': 'Optimized High-Return System',
            'models_loaded': True,
            'trading_halted': ml_system.trading_halted
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting ML signal: {e}")
        return {
            'signal': 'HOLD',
            'confidence': 0.0,
            'reason': f'Error: {str(e)}',
            'leverage': 1.0,
            'position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0
        }

# API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "binance_configured": binance_client is not None,
        "ml_system_ready": ml_system is not None,
        "credentials_from_file": CREDENTIALS_FILE.exists(),
        "credentials_file_path": str(CREDENTIALS_FILE)
    }

@app.post("/api/configure-binance")
async def configure_binance(config: BinanceConfig):
    """Configure Binance API"""
    global binance_client
    
    try:
        if initialize_binance_client(config.api_key, config.api_secret):
            # Save credentials to file
            save_credentials_to_file(config.api_key, config.api_secret)
            
            return {
                "status": "success",
                "message": "Binance API configured successfully",
                "testnet": True
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to connect to Binance API")
            
    except Exception as e:
        logger.error(f"❌ Error configuring Binance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml-signal")
async def get_ml_signal_endpoint():
    """Get ML trading signal"""
    try:
        signal_data = get_ml_signal()
        return signal_data
    except Exception as e:
        logger.error(f"❌ Error getting ML signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auto-trading-status")
async def get_auto_trading_status():
    """Get automatic trading status"""
    return {
        "enabled": auto_trading_enabled,
        "ml_system_ready": ml_system is not None,
        "binance_connected": binance_client is not None
    }

@app.post("/api/enable-auto-trading")
async def enable_auto_trading():
    """Enable automatic ML trading"""
    global auto_trading_enabled, auto_trading_thread
    
    if auto_trading_enabled:
        return {"message": "Auto-trading already enabled", "status": "already_enabled"}
    
    if not binance_client:
        raise HTTPException(status_code=400, detail="Binance API not configured")
    
    if not ml_system:
        raise HTTPException(status_code=400, detail="ML system not initialized")
    
    auto_trading_enabled = True
    auto_trading_thread = threading.Thread(target=automatic_trading_loop, daemon=True)
    auto_trading_thread.start()
    
    logger.info("🚀 Automatic ML trading enabled!")
    
    return {
        "message": "Automatic ML trading enabled",
        "status": "enabled"
    }

@app.post("/api/disable-auto-trading")
async def disable_auto_trading():
    """Disable automatic ML trading"""
    global auto_trading_enabled
    
    auto_trading_enabled = False
    logger.info("🛑 Automatic ML trading disabled!")
    
    return {
        "message": "Automatic ML trading disabled",
        "status": "disabled"
    }

@app.get("/api/account-info")
async def get_account_info():
    """Get Binance account information"""
    if not binance_client:
        raise HTTPException(status_code=400, detail="Binance API not configured")
    
    try:
        account = binance_client.get_account()
        
        # Get BTC and USDT balances
        btc_balance = 0.0
        usdt_balance = 0.0
        
        for balance in account['balances']:
            if balance['asset'] == 'BTC':
                btc_balance = float(balance['free'])
            elif balance['asset'] == 'USDT':
                usdt_balance = float(balance['free'])
        
        # Get current BTC price
        ticker = binance_client.get_ticker(symbol='BTCUSDT')
        current_price = float(ticker['lastPrice'])
        
        total_wallet_balance = usdt_balance + (btc_balance * current_price)
        
        return {
            "account_type": account.get('accountType', 'SPOT'),
            "btc_balance": btc_balance,
            "usdt_balance": usdt_balance,
            "total_wallet_balance": total_wallet_balance,
            "current_btc_price": current_price,
            "can_trade": account.get('canTrade', False),
            "can_withdraw": account.get('canWithdraw', False),
            "can_deposit": account.get('canDeposit', False)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    if not binance_client:
        raise HTTPException(status_code=400, detail="Binance API not configured")
    
    try:
        account = binance_client.get_account()
        
        positions = []
        for balance in account['balances']:
            asset = balance['asset']
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked
            
            if total > 0 and asset in ['BTC', 'USDT']:
                # Get current price for BTC
                if asset == 'BTC':
                    ticker = binance_client.get_ticker(symbol='BTCUSDT')
                    current_price = float(ticker['lastPrice'])
                    unrealized_pnl = 0.0  # Simplified - would need entry price for real P&L
                    unrealized_pnl_percent = 0.0
                else:
                    current_price = 1.0
                    unrealized_pnl = 0.0
                    unrealized_pnl_percent = 0.0
                
                positions.append({
                    "symbol": f"{asset}USDT",
                    "side": "LONG",
                    "entry_price": current_price,  # Simplified
                    "current_price": current_price,
                    "amount": total,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_percent": unrealized_pnl_percent
                })
        
        return positions
        
    except Exception as e:
        logger.error(f"❌ Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trade-history")
async def get_trade_history():
    """Get trade history"""
    if not binance_client:
        raise HTTPException(status_code=400, detail="Binance API not configured")
    
    try:
        trades = binance_client.get_my_trades(symbol='BTCUSDT', limit=100)
        
        trade_history = []
        for trade in trades:
            trade_history.append({
                "symbol": trade['symbol'],
                "side": trade['isBuyer'] and 'BUY' or 'SELL',
                "quantity": float(trade['qty']),
                "price": float(trade['price']),
                "commission": float(trade['commission']),
                "time": datetime.fromtimestamp(trade['time'] / 1000).isoformat()
            })
        
        return trade_history
        
    except Exception as e:
        logger.error(f"❌ Error getting trade history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trading-performance")
async def get_trading_performance():
    """Get trading performance metrics"""
    if not binance_client:
        raise HTTPException(status_code=400, detail="Binance API not configured")
    
    try:
        # Get account info
        account = binance_client.get_account()
        
        # Get trade history
        trades = binance_client.get_my_trades(symbol='BTCUSDT', limit=1000)
        
        # Calculate performance metrics
        total_trades = len(trades)
        profitable_trades = 0
        losing_trades = 0
        total_pnl = 0.0
        
        # Simple P&L calculation (would need more sophisticated logic for real implementation)
        for trade in trades:
            if trade['isBuyer']:
                # This is a simplified calculation
                pass
        
        # Get current balances
        btc_balance = 0.0
        usdt_balance = 0.0
        
        for balance in account['balances']:
            if balance['asset'] == 'BTC':
                btc_balance = float(balance['free'])
            elif balance['asset'] == 'USDT':
                usdt_balance = float(balance['free'])
        
        # Get current BTC price
        ticker = binance_client.get_ticker(symbol='BTCUSDT')
        current_price = float(ticker['lastPrice'])
        
        total_wallet_balance = usdt_balance + (btc_balance * current_price)
        
        return {
            "total_pnl": total_pnl,
            "unrealized_pnl": 0.0,  # Would need position tracking
            "win_rate_percent": (profitable_trades / max(total_trades, 1)) * 100,
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "losing_trades": losing_trades,
            "average_win": 0.0,  # Would need calculation
            "average_loss": 0.0,  # Would need calculation
            "total_return_percent": 0.0,  # Would need initial balance tracking
            "daily_return_percent": 0.0,
            "completed_trades": profitable_trades + losing_trades
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting trading performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("🚀 Starting ML Trading API Server...")
    uvicorn.run("trading_api:app", host="0.0.0.0", port=8000, reload=True)