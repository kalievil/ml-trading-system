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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import joblib

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
open_positions = {}  # Track open positions for stop-loss/take-profit

# Credentials file path
CREDENTIALS_FILE = Path("binance_credentials.json")

# Render outbound IP addresses (from Render dashboard)
# These IPs need to be whitelisted on Binance for the API to work
RENDER_IP_ADDRESSES = [
    "44.226.145.213",
    "54.187.200.255",
    "34.213.214.55",
    "35.164.95.156",
    "44.230.95.183",
    "44.229.200.200",
    "74.220.48.0/24",
    "74.220.56.0/24"
]

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

def get_current_server_ip():
    """Get the current server's public IP address
    
    Returns:
        str: Public IP address or None if unavailable
    """
    try:
        import urllib.request
        import urllib.error
        
        # Try multiple services for reliability
        services = [
            "https://api.ipify.org",
            "https://checkip.amazonaws.com",
            "https://ifconfig.me/ip"
        ]
        
        for service in services:
            try:
                with urllib.request.urlopen(service, timeout=5) as response:
                    ip = response.read().decode('utf-8').strip()
                    if ip and len(ip) > 0:
                        logger.info(f"🌐 Current server IP: {ip}")
                        return ip
            except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
                continue
        
        return None
    except Exception as e:
        logger.warning(f"⚠️ Could not determine server IP: {e}")
        return None

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
        error_str_lower = error_msg.lower()
        
        # Check for geo-restriction errors
        if "restricted location" in error_str_lower or "service unavailable" in error_str_lower:
            error_msg = (
                "Binance Testnet is blocking requests from this location (IP-based restriction). "
                "This is common with cloud hosting providers like Render. "
                "Solutions: 1) Run the API locally, 2) Use a VPS in an allowed region, "
                "3) Contact Binance support about IP whitelisting."
            )
        elif "eligibility" in error_str_lower or "terms" in error_str_lower:
            error_msg = (
                "Binance Testnet has geo-restrictions that block cloud hosting providers. "
                "The service is working correctly, but Binance is rejecting connections from Render's IP addresses. "
                "Consider: 1) Running locally, 2) Using a different hosting provider, "
                "3) Contacting Binance about IP restrictions."
            )
        
        logger.error(f"❌ Failed to initialize Binance client: {error_msg}")
        binance_client = None
        return False, error_msg

def initialize_ml_system():
    """Initialize ML system"""
    global ml_system
    try:
        ml_system = OptimizedHighReturnSystem(max_leverage=5)
        logger.info("✅ ML system initialized successfully")
        # Try to load saved scaler and models for live predictions
        try:
            models_path = Path("user_data/models")
            logger.info(f"📂 Looking for models in: {models_path.absolute()}")
            
            # Check if models directory exists
            if not models_path.exists():
                logger.warning(f"⚠️ Models directory does not exist: {models_path.absolute()}")
                logger.warning("⚠️ You need to train models first or copy model files to this directory")
            else:
                logger.info(f"✅ Models directory exists: {models_path.absolute()}")
            
            scaler_path = models_path / "scaler.pkl"
            if scaler_path.exists():
                ml_system.scaler = joblib.load(scaler_path)
                logger.info("🧪 Loaded scaler for live predictions")
            else:
                logger.warning(f"⚠️ Scaler not found: {scaler_path.absolute()}")
            
            # Load ensemble models if present
            model_files = {
                'xgboost_performance': models_path / 'xgboost_performance.pkl',
                'xgboost_aggressive': models_path / 'xgboost_aggressive.pkl',
                'xgboost_balanced': models_path / 'xgboost_balanced.pkl',
            }
            
            # Log which model files exist
            for name, path in model_files.items():
                if path.exists():
                    logger.info(f"✅ Found model file: {name} at {path.absolute()}")
                else:
                    logger.warning(f"⚠️ Model file not found: {name} at {path.absolute()}")
            
            # Check for ensemble weights file (saved by algorithm)
            ensemble_weights_path = models_path / "ensemble_weights.json"
            if ensemble_weights_path.exists():
                logger.info(f"✅ Found ensemble weights file: {ensemble_weights_path.absolute()}")
            else:
                logger.info(f"ℹ️ Ensemble weights file not found (will use equal weights): {ensemble_weights_path.absolute()}")
            
            # Load models FIRST, then apply compatibility patch
            loaded_models = {}
            for name, path in model_files.items():
                if path.exists():
                    try:
                        logger.info(f"🔄 Attempting to load model: {name} from {path.absolute()}")
                        model = joblib.load(path)
                        loaded_models[name] = model
                        logger.info(f"🧪 Loaded model: {name} successfully")
                    except Exception as load_error:
                        import traceback
                        logger.error(f"❌ Failed to load model {name}: {load_error}")
                        logger.error(f"❌ Traceback for {name}: {traceback.format_exc()}")
                        # Try to continue with other models
                        continue
            if loaded_models:
                ml_system.models = loaded_models
                
                # Fix XGBoost version compatibility: add missing attributes to model instances
                # Apply patch AFTER loading models to avoid interfering with deserialization
                try:
                    import xgboost as xgb
                    
                    # Compatibility attributes with default values
                    COMPAT_ATTRS = {
                        'use_label_encoder': False,
                        'gpu_id': None,
                        'tree_method': 'hist'
                    }
                    
                    # Patch each loaded model instance by adding missing attributes
                    for name, model in loaded_models.items():
                        for attr_name, default_value in COMPAT_ATTRS.items():
                            if not hasattr(model, attr_name):
                                setattr(model, attr_name, default_value)
                    
                    logger.info("🔧 Applied XGBoost compatibility patch (use_label_encoder, gpu_id, tree_method)")
                except Exception as patch_error:
                    logger.warning(f"⚠️ Could not apply XGBoost patch: {patch_error}")
                
                # Load ensemble weights from file if available (same logic as algorithm)
                ensemble_weights_path = models_path / "ensemble_weights.json"
                if ensemble_weights_path.exists():
                    try:
                        with open(ensemble_weights_path, 'r') as f:
                            saved_weights = json.load(f)
                        # Filter to only weights for models we loaded
                        ml_system.ensemble_weights = {k: saved_weights.get(k, 0) for k in loaded_models.keys()}
                        # Normalize weights to sum to 1 (same as algorithm logic)
                        total_weight = sum(ml_system.ensemble_weights.values())
                        if total_weight > 0:
                            ml_system.ensemble_weights = {k: v/total_weight for k, v in ml_system.ensemble_weights.items()}
                        else:
                            # Fallback to equal weights if normalization fails
                            weight = 1.0 / len(loaded_models)
                            ml_system.ensemble_weights = {k: weight for k in loaded_models.keys()}
                        logger.info(f"🧪 Loaded ensemble weights from file: {ml_system.ensemble_weights}")
                    except Exception as weight_error:
                        logger.warning(f"⚠️ Could not load ensemble weights: {weight_error}")
                        # Fallback to equal weights (same as algorithm default)
                        weight = 1.0 / len(loaded_models)
                        ml_system.ensemble_weights = {k: weight for k in loaded_models.keys()}
                        logger.info(f"🧪 Using equal ensemble weights: {ml_system.ensemble_weights}")
                else:
                    # If weights file doesn't exist, use equal weights (same as algorithm default)
                    weight = 1.0 / len(loaded_models)
                    ml_system.ensemble_weights = {k: weight for k in loaded_models.keys()}
                    logger.info(f"🧪 Ensemble weights file not found, using equal weights: {ml_system.ensemble_weights}")
                
                logger.info(f"🧪 Ensemble ready with {len(loaded_models)} models and weights: {ml_system.ensemble_weights}")
            else:
                logger.error("❌ No models were loaded! Check that model .pkl files exist in user_data/models/")
                logger.error("   Required files: xgboost_performance.pkl, xgboost_aggressive.pkl, xgboost_balanced.pkl")
        except Exception as model_e:
            logger.error(f"❌ Could not load saved models/scaler: {model_e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
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

def check_position_management_sync():
    """Check and manage open positions for stop-loss/take-profit (sync version)"""
    global open_positions, binance_client
    
    if not binance_client or not open_positions:
        return
    
    try:
        # Get current BTC price
        ticker = binance_client.get_ticker(symbol='BTCUSDT')
        current_price = float(ticker['lastPrice'])
        
        positions_to_close = []
        
        for position_id, position in open_positions.items():
            entry_price = position['entry_price']
            side = position['side']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            should_close = False
            close_reason = ""
            
            if side == 'BUY':
                # Long position
                if current_price <= stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif current_price >= take_profit:
                    should_close = True
                    close_reason = "take_profit"
            else:
                # Short position (if implemented)
                if current_price >= stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif current_price <= take_profit:
                    should_close = True
                    close_reason = "take_profit"
            
            if should_close:
                positions_to_close.append((position_id, close_reason))
        
        # Close positions that hit stop-loss or take-profit
        for position_id, reason in positions_to_close:
            close_position_sync(position_id, reason)
            
    except Exception as e:
        logger.error(f"❌ Position management error: {e}")

def close_position_sync(position_id, reason):
    """Close a specific position (sync version)"""
    global open_positions, binance_client
    
    if position_id not in open_positions:
        return
    
    position = open_positions[position_id]
    
    try:
        # Get current account balance
        account = binance_client.get_account()
        btc_balance = 0.0
        
        for balance in account['balances']:
            if balance['asset'] == 'BTC':
                btc_balance = float(balance['free'])
                break
        
        if btc_balance > 0:
            # Execute SELL order
            order = binance_client.order_market_sell(
                symbol='BTCUSDT',
                quantity=f"{btc_balance:.6f}"
            )
            
            logger.info(f"✅ Position closed: {position_id} - {reason} - {btc_balance:.6f} BTC")
            
            # Remove from tracking
            del open_positions[position_id]
            
    except Exception as e:
        logger.error(f"❌ Error closing position {position_id}: {e}")

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
            
            # Check for position management first (stop-loss/take-profit)
            # Note: This is a sync function, so we'll call it directly
            check_position_management_sync()
            
            # Get ML signal
            signal_data = get_ml_signal()
            
            # Check if signal meets criteria for automatic execution
            if signal_data['confidence'] >= 0.63 and signal_data['signal'] != 'HOLD':
                logger.info(f"🎯 Auto-trading: {signal_data['signal']} signal with {signal_data['confidence']:.1%} confidence")
                
                try:
                    # Get current account balance
                    account = binance_client.get_account()
                    usdt_balance = 0.0
                    btc_balance = 0.0
                    
                    for balance in account['balances']:
                        if balance['asset'] == 'USDT':
                            usdt_balance = float(balance['free'])
                        elif balance['asset'] == 'BTC':
                            btc_balance = float(balance['free'])
                    
                    # Get current BTC price
                    ticker = binance_client.get_ticker(symbol='BTCUSDT')
                    current_price = float(ticker['lastPrice'])
                    
                    # Execute trade based on signal
                    if signal_data['signal'] == 'BUY':
                        # FIXED: Only use available USDT, not total portfolio value
                        usdt_to_spend = usdt_balance * signal_data['position_size']
                        
                        # Check minimum order size (Binance minimum is typically 10 USDT)
                        if usdt_to_spend < 10:
                            logger.warning(f"⚠️ Order size too small: {usdt_to_spend:.2f} USDT (minimum 10 USDT)")
                            time.sleep(10)
                            continue
                        
                        # Calculate BTC quantity from USDT amount
                        btc_quantity = usdt_to_spend / current_price
                        
                        # Execute BUY order
                        order = binance_client.order_market_buy(
                            symbol='BTCUSDT',
                            quoteOrderQty=f"{usdt_to_spend:.2f}"  # Use quoteOrderQty (USDT amount) instead of quantity
                        )
                        logger.info(f"✅ Auto-executed BUY order: {order['orderId']} - {usdt_to_spend:.2f} USDT (~{btc_quantity:.6f} BTC)")
                        
                        # Track the position for stop-loss/take-profit management
                        position_id = f"pos_{int(time.time())}"
                        stop_loss_price = current_price * (1 - signal_data.get('stop_loss_pct', 0.018))  # 1.8% stop-loss
                        take_profit_price = current_price * (1 + signal_data.get('take_profit_pct', 0.015))  # 1.5% take-profit
                        
                        open_positions[position_id] = {
                            'side': 'BUY',
                            'entry_price': current_price,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'entry_time': datetime.now(),
                            'order_id': order['orderId']
                        }
                        
                        logger.info(f"📊 Position tracked: {position_id} - Entry: {current_price:.2f}, SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}")
                    
                    elif signal_data['signal'] == 'SELL':
                        # For SELL, position_size is percentage of BTC holdings
                        btc_to_sell = btc_balance * signal_data['position_size']
                        
                        # Ensure we have BTC to sell
                        if btc_to_sell > btc_balance:
                            logger.warning(f"⚠️ Insufficient BTC. Need {btc_to_sell:.6f} BTC, have {btc_balance:.6f} BTC")
                            btc_to_sell = btc_balance * 0.99  # Use 99% to leave some
                        
                        # Check minimum order size
                        if btc_to_sell < 0.00001:
                            logger.warning(f"⚠️ Order size too small: {btc_to_sell:.6f} BTC (minimum 0.00001 BTC)")
                            time.sleep(10)
                            continue
                        
                        # Execute SELL order
                        order = binance_client.order_market_sell(
                            symbol='BTCUSDT',
                            quantity=f"{btc_to_sell:.6f}"
                        )
                        logger.info(f"✅ Auto-executed SELL order: {order['orderId']} - {btc_to_sell:.6f} BTC (~{btc_to_sell * current_price:.2f} USDT)")
                    
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
        
        # Fetch live klines to build features matching the strategy
        klines = binance_client.get_klines(symbol='BTCUSDT', interval='5m', limit=max(ml_system.lookback_period + 50, 300))
        if not klines:
            raise RuntimeError("No klines returned from Binance")
        # Build dataframe
        df = pd.DataFrame(klines, columns=[
            'open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base','taker_quote','ignore'
        ])
        df = df[['open_time','open','high','low','close','volume']].copy()
        # Ensure only one timestamp column exists later when renaming
        # Create a temporary column to avoid duplicate 'timestamp' labels
        df['kline_timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        for col in ['open','high','low','close','volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        # Use the strategy feature engineering
        # Build dataframe with a proper datetime 'timestamp' column expected by the feature builder
        df_renamed = df.copy()
        df_renamed['timestamp'] = df_renamed['kline_timestamp']  # ensure datetime dtype
        # Drop helper/original columns to avoid confusion
        drop_cols = [c for c in ['kline_timestamp', 'open_time'] if c in df_renamed.columns]
        if drop_cols:
            df_renamed = df_renamed.drop(columns=drop_cols)
        df_features = ml_system.create_features(df_renamed)
        if df_features is None or len(df_features) == 0:
            raise RuntimeError("Feature creation returned empty dataframe")
        # Extract latest feature row
        feature_columns = [c for c in df_features.columns if c not in ['timestamp','target','open','high','low','close','volume']]
        latest_row = df_features.iloc[-1]
        X_row = latest_row[feature_columns].values.reshape(1, -1)
        # Scale
        if not hasattr(ml_system, 'scaler') or ml_system.scaler is None:
            raise RuntimeError("Scaler not loaded. Train and save models first or provide scaler.pkl")
        X_scaled = ml_system.scaler.transform(X_row)
        # Predict via ensemble (same logic as algorithm's _ensemble_predict)
        if not hasattr(ml_system, 'models') or not ml_system.models:
            raise RuntimeError("Models not loaded. Train and save models first or provide model pkl files")
        # Collect all probabilities first (same as algorithm)
        all_probabilities = []
        for name, model in ml_system.models.items():
            prob = model.predict_proba(X_scaled)
            all_probabilities.append(prob)
        # Weight probabilities using ensemble weights (same logic as algorithm)
        ensemble_probabilities = np.zeros_like(all_probabilities[0], dtype=float)
        for i, (name, prob) in enumerate(zip(ml_system.models.keys(), all_probabilities)):
            weight = ml_system.ensemble_weights.get(name, 1.0/len(ml_system.models))
            ensemble_probabilities += weight * prob
        # Determine class: 0=HOLD, 1=BUY, 2=SELL (same as algorithm)
        cls_idx = int(np.argmax(ensemble_probabilities, axis=1)[0])
        confidence = float(np.max(ensemble_probabilities, axis=1)[0])
        signal_map = {0:'HOLD', 1:'BUY', 2:'SELL'}
        signal = signal_map.get(cls_idx, 'HOLD')
        # Volatility from features
        volatility = float(latest_row.get('volatility', 0.02))
        # Current price
        current_price = float(df['close'].iloc[-1])
        # Calculate position size and risk parameters from algorithm
        position_size = float(ml_system.calculate_position_size(confidence, volatility))
        leverage = float(ml_system.calculate_dynamic_leverage(confidence, volatility))
        stop_loss = current_price * (1 - ml_system.stop_loss_pct)
        take_profit = current_price * (1 + ml_system.take_profit_pct)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': f'Ensemble ML prediction with {confidence:.1%} confidence',
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

@app.get("/")
async def root():
    """Root endpoint - redirects to API documentation"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

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

@app.get("/api/server-ips")
async def get_server_ips():
    """Get Render server IP addresses for Binance whitelisting
    
    Returns the IP addresses that need to be whitelisted on Binance
    for the API to work from Render.
    """
    current_ip = get_current_server_ip()
    
    return {
        "render_ip_addresses": RENDER_IP_ADDRESSES,
        "current_server_ip": current_ip,
        "instructions": {
            "step_1": "Go to https://testnet.binancefuture.com/en/my/settings/api-management",
            "step_2": "Edit your API key",
            "step_3": "Enable 'Restrict access to trusted IPs only' (if available)",
            "step_4": "Add all IP addresses listed in 'render_ip_addresses'",
            "step_5": "Save and try connecting again",
            "note": "You may need to add individual IPs (not CIDR ranges) depending on Binance's interface"
        },
        "individual_ips": [
            ip for ip in RENDER_IP_ADDRESSES if "/" not in ip
        ],
        "ip_ranges": [
            ip for ip in RENDER_IP_ADDRESSES if "/" in ip
        ],
        "testnet_url": "https://testnet.binancefuture.com/en/my/settings/api-management"
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
        elif "restricted location" in error_detail or "service unavailable" in error_detail or "eligibility" in error_detail:
            # Get current server IP for whitelisting instructions
            current_ip = get_current_server_ip()
            ip_list = ", ".join(RENDER_IP_ADDRESSES[:6])  # Show individual IPs
            
            error_detail = (
                "⚠️ Binance Geo-Restriction: Binance Testnet is blocking requests from Render's IP addresses.\n\n"
                "**Solution: Whitelist Render IPs on Binance:**\n"
                "1. Go to https://testnet.binancefuture.com/en/my/settings/api-management\n"
                "2. Edit your API key\n"
                "3. Enable 'Restrict access to trusted IPs only'\n"
                "4. Add these Render IP addresses:\n"
            )
            error_detail += f"   {ip_list}\n"
            error_detail += "   (Full list available at: /api/server-ips)\n\n"
            
            if current_ip:
                error_detail += f"**Current server IP detected:** {current_ip}\n"
                error_detail += "Make sure this IP is whitelisted on Binance.\n\n"
            
            error_detail += (
                "**Alternative Solutions:**\n"
                "- Run the API locally on your machine (works perfectly)\n"
                "- Use a VPS in an allowed region (AWS, DigitalOcean, etc.)\n"
                "- Contact Binance support if IP whitelisting doesn't resolve the issue\n"
            )
        
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

@app.get("/api/open-positions")
async def get_open_positions():
    """Get tracked open positions with stop-loss/take-profit info"""
    global open_positions
    
    if not binance_client:
        raise HTTPException(status_code=400, detail="Binance API not configured")
    
    try:
        # Get current BTC price
        ticker = binance_client.get_ticker(symbol='BTCUSDT')
        current_price = float(ticker['lastPrice'])
        
        tracked_positions = []
        for position_id, position in open_positions.items():
            entry_price = position['entry_price']
            current_pnl = (current_price - entry_price) / entry_price * 100
            
            tracked_positions.append({
                "position_id": position_id,
                "side": position['side'],
                "entry_price": entry_price,
                "current_price": current_price,
                "stop_loss": position['stop_loss'],
                "take_profit": position['take_profit'],
                "unrealized_pnl_percent": current_pnl,
                "entry_time": position['entry_time'].isoformat(),
                "order_id": position['order_id']
            })
        
        return {
            "tracked_positions": tracked_positions,
            "total_positions": len(tracked_positions)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting open positions: {e}")
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

# Pydantic models for manual trading
class ManualTradeRequest(BaseModel):
    side: str = Field(..., description="BUY or SELL")
    amount_usdt: Optional[float] = Field(None, description="Amount in USDT (for BUY)")
    amount_btc: Optional[float] = Field(None, description="Amount in BTC (for SELL)")
    percentage: Optional[float] = Field(None, description="Percentage of available balance (0.1 = 10%)")

class ConvertRequest(BaseModel):
    percent: float = Field(0.25, description="Percentage of BTC to convert to USDT (0.25 = 25%)")

@app.post("/api/manual-trade")
async def manual_trade(request: ManualTradeRequest):
    """Execute manual buy or sell order"""
    if not binance_client:
        raise HTTPException(status_code=400, detail="Binance API not configured")
    
    try:
        # Get current account info
        account = binance_client.get_account()
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
        
        if request.side.upper() == 'BUY':
            # Calculate USDT amount to spend
            if request.percentage:
                usdt_to_spend = usdt_balance * request.percentage
            elif request.amount_usdt:
                usdt_to_spend = request.amount_usdt
            else:
                raise HTTPException(status_code=400, detail="Must specify either amount_usdt or percentage for BUY")
            
            # Validate order size
            if usdt_to_spend < 10:
                raise HTTPException(status_code=400, detail=f"Order too small: {usdt_to_spend:.2f} USDT (minimum 10 USDT)")
            
            if usdt_to_spend > usdt_balance:
                raise HTTPException(status_code=400, detail=f"Insufficient USDT. Need {usdt_to_spend:.2f}, have {usdt_balance:.2f}")
            
            # Execute BUY order
            order = binance_client.order_market_buy(
                symbol='BTCUSDT',
                quoteOrderQty=f"{usdt_to_spend:.2f}"
            )
            
            btc_quantity = usdt_to_spend / current_price
            logger.info(f"✅ Manual BUY executed: {order['orderId']} - {usdt_to_spend:.2f} USDT (~{btc_quantity:.6f} BTC)")
            
            return {
                "status": "success",
                "action": "BUY",
                "order_id": order['orderId'],
                "usdt_spent": usdt_to_spend,
                "btc_received": btc_quantity,
                "price": current_price
            }
        
        elif request.side.upper() == 'SELL':
            # Calculate BTC amount to sell
            if request.percentage:
                btc_to_sell = btc_balance * request.percentage
            elif request.amount_btc:
                btc_to_sell = request.amount_btc
            else:
                raise HTTPException(status_code=400, detail="Must specify either amount_btc or percentage for SELL")
            
            # Validate order size
            if btc_to_sell < 0.00001:
                raise HTTPException(status_code=400, detail=f"Order too small: {btc_to_sell:.6f} BTC (minimum 0.00001 BTC)")
            
            if btc_to_sell > btc_balance:
                raise HTTPException(status_code=400, detail=f"Insufficient BTC. Need {btc_to_sell:.6f}, have {btc_balance:.6f}")
            
            # Execute SELL order
            order = binance_client.order_market_sell(
                symbol='BTCUSDT',
                quantity=f"{btc_to_sell:.6f}"
            )
            
            usdt_received = btc_to_sell * current_price
            logger.info(f"✅ Manual SELL executed: {order['orderId']} - {btc_to_sell:.6f} BTC (~{usdt_received:.2f} USDT)")
            
            return {
                "status": "success",
                "action": "SELL",
                "order_id": order['orderId'],
                "btc_sold": btc_to_sell,
                "usdt_received": usdt_received,
                "price": current_price
            }
        
        else:
            raise HTTPException(status_code=400, detail="Side must be 'BUY' or 'SELL'")
    
    except Exception as e:
        logger.error(f"❌ Manual trade failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/convert-btc-to-usdt")
async def convert_btc_to_usdt(request: ConvertRequest):
    """Convert BTC to USDT (useful for getting trading funds)"""
    if not binance_client:
        raise HTTPException(status_code=400, detail="Binance API not configured")

    try:
        account = binance_client.get_account()
        btc_balance = 0.0
        for balance in account['balances']:
            if balance['asset'] == 'BTC':
                btc_balance = float(balance['free'])
                break

        if btc_balance <= 0:
            raise HTTPException(status_code=400, detail="No BTC balance available")

        btc_to_sell = max(btc_balance * max(min(request.percent, 1.0), 0.0), 0.00002)  # ensure above min
        ticker = binance_client.get_ticker(symbol='BTCUSDT')
        current_price = float(ticker['lastPrice'])
        if btc_to_sell * current_price < 10:
            raise HTTPException(status_code=400, detail="Order would be < 10 USDT minimum")

        order = binance_client.order_market_sell(
            symbol='BTCUSDT',
            quantity=f"{btc_to_sell:.6f}"
        )
        return {
            "status": "success", 
            "sold_btc": btc_to_sell, 
            "approx_usdt": btc_to_sell * current_price, 
            "order_id": order['orderId']
        }
    except Exception as e:
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
    # Get port from environment variable (required for Render)
    # Default to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    
    # Only enable reload in development (not in production)
    # Render sets RENDER env var, we can use that or disable reload entirely for production
    reload = os.environ.get("ENVIRONMENT") != "production" and os.environ.get("RENDER") != "true"
    
    logger.info(f"🚀 Starting ML Trading API Server on port {port}...")
    logger.info(f"📝 Reload enabled: {reload}")
    
    uvicorn.run("trading_api:app", host="0.0.0.0", port=port, reload=reload)