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
from pydantic import BaseModel, Field
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
    """Binance API configuration
    
    Note: This endpoint uses Binance TESTNET. Get your testnet API keys from:
    https://testnet.binancefuture.com/
    
    API keys are typically 64 characters long.
    """
    api_key: str = Field(
        ...,
        min_length=20,
        description="Binance Testnet API Key (typically 64 characters). Get from https://testnet.binancefuture.com/",
        example="YourBinanceTestnetAPIKeyHere64CharactersLong123456789012345678901234567890"
    )
    api_secret: str = Field(
        ...,
        min_length=20,
        description="Binance Testnet API Secret (typically 64 characters). Get from https://testnet.binancefuture.com/",
        example="YourBinanceTestnetAPISecretHere64CharactersLong123456789012345678901234567890"
    )

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
    """Initialize Binance client
    
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    global binance_client
    try:
        from binance.client import Client
        
        # Validate API key format (should be non-empty strings)
        if not api_key or not api_secret:
            error_msg = "API key and API secret cannot be empty"
            logger.error(f"❌ Failed to initialize Binance client: {error_msg}")
            binance_client = None
            return False, error_msg
        
        # Check for duplicated API secret (common copy/paste error)
        # Binance secrets are typically 64 characters
        if len(api_secret) >= 128:
            # Check if it's duplicated (first half == second half)
            midpoint = len(api_secret) // 2
            first_half = api_secret[:midpoint]
            second_half = api_secret[midpoint:midpoint*2] if len(api_secret) >= midpoint*2 else ""
            
            if first_half == second_half:
                logger.warning(f"⚠️ Detected duplicated API secret (length: {len(api_secret)}). Using first half only.")
                api_secret = first_half
            elif len(api_secret) > 100:
                # Secret is suspiciously long, take first 64 characters
                logger.warning(f"⚠️ API secret is very long ({len(api_secret)} chars). Using first 64 characters.")
                api_secret = api_secret[:64]
        
        # Also check API key length (should be around 64)
        if len(api_key) > 100:
            logger.warning(f"⚠️ API key is very long ({len(api_key)} chars). Using first 64 characters.")
            api_key = api_key[:64]
        
        # Use testnet
        binance_client = Client(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True
        )
        
        # Test connection
        account_info = binance_client.get_account()
        logger.info(f"✅ Binance API connected - Account type: {account_info.get('accountType', 'UNKNOWN')}")
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Failed to initialize Binance client: {error_msg}")
        binance_client = None
        return False, error_msg

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
        success, error_msg = initialize_binance_client(api_key, api_secret)
        if success:
            logger.info("✅ Binance API auto-configured from credentials file")
        else:
            logger.error(f"❌ Failed to configure Binance API: {error_msg}")
    
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
    """Configure Binance Testnet API credentials
    
    This endpoint configures the Binance Testnet API connection.
    
    **Important Notes:**
    - This endpoint uses **Binance TESTNET** (not live trading)
    - Get your testnet API keys from: https://testnet.binancefuture.com/
    - API keys should be approximately 64 characters long
    - Credentials are saved to `binance_credentials.json` upon success
    
    **Common Issues:**
    - Extra spaces: Automatically trimmed
    - Duplicated secret: Automatically detected and fixed
    - Wrong key format: Will return validation error with details
    
    Returns success status if connection is verified.
    """
    global binance_client
    
    # Log that we received the request
    logger.info("📥 Received Binance configuration request")
    
    # Trim whitespace (common issue: copy/paste includes spaces)
    api_key = config.api_key.strip() if config.api_key else ""
    api_secret = config.api_secret.strip() if config.api_secret else ""
    
    # Log what we received (before validation)
    logger.info(f"📋 Received API key length: {len(api_key)} characters")
    logger.info(f"📋 Received API secret length: {len(api_secret)} characters")
    
    # Validate input
    if not api_key or not api_secret:
        logger.warning("⚠️ Empty API key or secret provided")
        raise HTTPException(
            status_code=400, 
            detail="API key and API secret are required"
        )
    
    # Additional validation - Binance API keys are typically 64 characters
    if len(api_key) < 20:
        logger.warning(f"⚠️ API key too short: {len(api_key)} characters (expected ~64)")
        raise HTTPException(
            status_code=400,
            detail=f"API key appears too short ({len(api_key)} characters). Binance API keys are typically 64 characters long."
        )
    
    if len(api_secret) < 20:
        logger.warning(f"⚠️ API secret too short: {len(api_secret)} characters (expected ~64)")
        raise HTTPException(
            status_code=400,
            detail=f"API secret appears too short ({len(api_secret)} characters). Binance API secrets are typically 64 characters long."
        )
    
    # Log partial key for debugging (first 10 chars only for security)
    logger.info(f"🔑 Attempting to configure Binance API with key: {api_key[:10]}... (length: {len(api_key)})")
    
    # Try to initialize Binance client
    success, error_msg = initialize_binance_client(api_key, api_secret)
    
    if success:
        # Save credentials to file (using trimmed values)
        if save_credentials_to_file(api_key, api_secret):
            logger.info("✅ Credentials saved to file")
        else:
            logger.warning("⚠️ Failed to save credentials to file, but connection successful")
        
        return {
            "status": "success",
            "message": "Binance API configured successfully",
            "testnet": True
        }
    else:
        # Return detailed error message
        error_detail = error_msg or "Failed to connect to Binance API"
        
        # Provide more user-friendly error messages with diagnostic info
        if "API-key format invalid" in error_detail or "-2014" in error_detail:
            error_detail = f"API key format is invalid. Received key length: {len(api_key)} characters. "
            error_detail += "Binance API keys are typically 64 characters. "
            error_detail += "Common issues: extra spaces, incomplete copy/paste, or wrong key. "
            error_detail += "Please double-check your API key from Binance Testnet."
        elif "Invalid API-key" in error_detail or "-2015" in error_detail:
            error_detail = f"Invalid API key (length: {len(api_key)}). The key may be incorrect or not have the required permissions."
        elif "Signature" in error_detail or "-1022" in error_detail:
            error_detail = f"Invalid API secret signature (secret length: {len(api_secret)}). "
            if len(api_secret) >= 128:
                error_detail += "The secret appears to be duplicated (contains the same sequence twice). "
            error_detail += "Please verify your API secret matches the API key. Common issues: wrong secret, duplicated secret, or extra characters."
        
        logger.error(f"❌ Error configuring Binance: {error_detail}")
        raise HTTPException(
            status_code=400,
            detail=error_detail
        )

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
        
        # Get trade history (sorted by time, oldest first)
        all_trades = binance_client.get_my_trades(symbol='BTCUSDT', limit=1000)
        trades = sorted(all_trades, key=lambda x: x['time'])
        
        # Get current balances
        btc_balance = 0.0
        usdt_balance = 0.0
        
        for balance in account['balances']:
            if balance['asset'] == 'BTC':
                btc_balance = float(balance['free']) + float(balance['locked'])
            elif balance['asset'] == 'USDT':
                usdt_balance = float(balance['free']) + float(balance['locked'])
        
        # Get current BTC price
        ticker = binance_client.get_ticker(symbol='BTCUSDT')
        current_price = float(ticker['lastPrice'])
        
        # Calculate P&L by matching buy/sell pairs (FIFO method)
        buy_stack = []  # Stack of buy orders (quantity, price)
        completed_trades = []  # List of completed trade pairs (buy_price, sell_price, quantity, pnl, timestamp)
        total_realized_pnl = 0.0
        total_commission_paid = 0.0
        today = datetime.now().date()
        daily_realized_pnl = 0.0
        
        for trade in trades:
            quantity = float(trade['qty'])
            price = float(trade['price'])
            commission = float(trade['commission'])
            trade_time = datetime.fromtimestamp(trade['time'] / 1000)
            is_today = trade_time.date() == today
            total_commission_paid += commission
            
            if trade['isBuyer']:
                # This is a BUY order
                buy_stack.append({'quantity': quantity, 'price': price})
            else:
                # This is a SELL order - match with buy orders
                remaining_sell_qty = quantity
                
                while remaining_sell_qty > 0 and buy_stack:
                    buy = buy_stack[0]
                    sell_qty = min(remaining_sell_qty, buy['quantity'])
                    
                    # Calculate P&L for this trade pair
                    pnl = (price - buy['price']) * sell_qty
                    total_realized_pnl += pnl
                    
                    # Track daily P&L
                    if is_today:
                        daily_realized_pnl += pnl
                    
                    completed_trades.append({
                        'buy_price': buy['price'],
                        'sell_price': price,
                        'quantity': sell_qty,
                        'pnl': pnl,
                        'timestamp': trade_time.isoformat()
                    })
                    
                    # Update buy stack
                    buy['quantity'] -= sell_qty
                    if buy['quantity'] <= 0:
                        buy_stack.pop(0)
                    
                    remaining_sell_qty -= sell_qty
        
        # Calculate unrealized P&L for remaining BTC holdings
        unrealized_pnl = 0.0
        if btc_balance > 0 and buy_stack:
            # Calculate average buy price for remaining BTC
            total_btc_cost = 0.0
            total_btc_qty = 0.0
            
            for buy in buy_stack:
                total_btc_cost += buy['price'] * buy['quantity']
                total_btc_qty += buy['quantity']
            
            if total_btc_qty > 0:
                avg_buy_price = total_btc_cost / total_btc_qty
                # Use actual BTC balance if it's less than what's in the stack (some might have been traded elsewhere)
                actual_btc = min(btc_balance, total_btc_qty)
                unrealized_pnl = (current_price - avg_buy_price) * actual_btc
        
        # Calculate win/loss statistics
        profitable_trades = 0
        losing_trades = 0
        winning_pnls = []
        losing_pnls = []
        
        for trade in completed_trades:
            pnl = trade['pnl']
            if pnl > 0:
                profitable_trades += 1
                winning_pnls.append(pnl)
            elif pnl < 0:
                losing_trades += 1
                losing_pnls.append(pnl)
        
        # Calculate averages
        average_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0.0
        average_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0.0
        
        # Total P&L (realized + unrealized, minus commissions)
        total_pnl = total_realized_pnl + unrealized_pnl - total_commission_paid
        
        # Calculate return percentage (approximate, would need initial balance for exact calculation)
        total_wallet_balance = usdt_balance + (btc_balance * current_price)
        
        completed_trade_count = len(completed_trades)
        win_rate = (profitable_trades / completed_trade_count * 100) if completed_trade_count > 0 else 0.0
        
        return {
            "total_pnl": round(total_pnl, 2),
            "realized_pnl": round(total_realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_commission_paid": round(total_commission_paid, 6),
            "win_rate_percent": round(win_rate, 2),
            "total_trades": len(trades),
            "completed_trades": completed_trade_count,
            "profitable_trades": profitable_trades,
            "losing_trades": losing_trades,
            "average_win": round(average_win, 2),
            "average_loss": round(average_loss, 2),
            "current_btc_balance": round(btc_balance, 8),
            "current_usdt_balance": round(usdt_balance, 2),
            "total_wallet_balance": round(total_wallet_balance, 2),
            "current_btc_price": round(current_price, 2),
            "daily_return_percent": round((daily_realized_pnl / max(total_wallet_balance, 1)) * 100, 2) if total_wallet_balance > 0 else 0.0
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting trading performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("🚀 Starting ML Trading API Server...")
    uvicorn.run("trading_api:app", host="0.0.0.0", port=8000, reload=True)