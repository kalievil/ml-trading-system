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

# Import your ML algorithm - NOW USING TARGETED LONG LOSS FIX SYSTEM
from importlib import import_module
import sys
from pathlib import Path

# Import the TargetedLongLossFixSystem from 97%WR-AND-2940%.py
# We need to handle the special filename (starts with a number)
import importlib.util
algorithm_file_path = Path("97%WR-AND-2940%.py")

if algorithm_file_path.exists():
    spec = importlib.util.spec_from_file_location("ml_algorithm", algorithm_file_path)
    ml_algorithm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml_algorithm_module)
    TargetedLongLossFixSystem = ml_algorithm_module.TargetedLongLossFixSystem
else:
    # Fallback to old system if file doesn't exist
    from complete_high_return_optimized import OptimizedHighReturnSystem
    TargetedLongLossFixSystem = OptimizedHighReturnSystem

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
position_closing_in_progress = False  # Lock to prevent multiple simultaneous close operations

# BACKTEST CLONE: Step size tracking (1 barre sur 6 comme backtest)
last_evaluated_bar_time = None  # Track last bar timestamp we evaluated
step_size = 6  # Sample every 6th bar (same as backtest) - reduces frequency by 83%

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
        ml_system = TargetedLongLossFixSystem()
        logger.info("✅ ML system initialized successfully (Targeted Long Loss Fix System)")
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
            
            # Load selected features if present (CRITICAL for feature matching)
            selected_features_path = models_path / "selected_features.json"
            if selected_features_path.exists():
                try:
                    with open(selected_features_path, 'r') as f:
                        loaded_features = json.load(f)
                        # Ensure it's a list (not a dict or other structure)
                        if isinstance(loaded_features, list):
                            ml_system.selected_features = loaded_features
                        else:
                            logger.warning(f"⚠️ selected_features.json is not a list, ignoring")
                            ml_system.selected_features = None
                    if ml_system.selected_features:
                        logger.info(f"✅ Loaded {len(ml_system.selected_features)} selected features from file")
                        logger.info(f"   First 5 features: {ml_system.selected_features[:5]}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load selected features: {e}")
                    ml_system.selected_features = None
            else:
                logger.warning(f"⚠️ selected_features.json not found at {selected_features_path.absolute()}")
                logger.warning("⚠️ API will use all features (may cause errors if models were trained with subset)")
                ml_system.selected_features = None
            
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
                    # These attributes may be missing in models trained with older XGBoost versions
                    COMPAT_ATTRS = {
                        'use_label_encoder': False,
                        'gpu_id': None,
                        'tree_method': 'hist',
                        'predictor': None  # Added to fix AttributeError during predict_proba
                    }
                    
                    # Patch each loaded model instance by adding missing attributes
                    for name, model in loaded_models.items():
                        for attr_name, default_value in COMPAT_ATTRS.items():
                            if not hasattr(model, attr_name):
                                setattr(model, attr_name, default_value)
                                logger.debug(f"  ✓ Added missing attribute '{attr_name}' to {name}")
                    
                    logger.info(f"🔧 Applied XGBoost compatibility patch for {len(COMPAT_ATTRS)} attributes")
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
    logger.info("=" * 100)
    logger.info("🎯 BACKTEST CLONE MODE ENABLED")
    logger.info(f"   Step Size: {step_size} (evaluates 1 bar out of {step_size} - 83% reduction)")
    logger.info(f"   Frequency: ~1 evaluation every {step_size * 5} minutes (matches backtest)")
    logger.info("   TP/SL Detection: HIGH/LOW of bar (not just lastPrice)")
    logger.info("   Execution: Bar close price (not current market price)")
    logger.info("   Filters: Confidence threshold + _should_allow_long_signal (exact backtest match)")
    logger.info("   Position Sizing: Capital variable (compounding allowed)")
    logger.info("=" * 100)
    
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
        if ml_system:
            logger.info(f"   Long Confidence Threshold: {ml_system.long_confidence_threshold:.2f}")
            logger.info(f"   Long Stop Loss: {ml_system.long_stop_loss_pct*100:.2f}%")
            logger.info(f"   Long Take Profit: {ml_system.long_take_profit_base*100:.2f}%-{ml_system.long_take_profit_max*100:.2f}%")
            logger.info(f"   Position Size: {ml_system.base_position_size*100:.0f}%-{ml_system.max_position_size*100:.0f}%")
    else:
        logger.error("❌ Failed to initialize ML system")
    
    logger.info("🌐 API Server ready!")

def check_position_management_sync():
    """Check and manage open positions for stop-loss/take-profit (sync version)
    
    BACKTEST CLONE: Uses HIGH/LOW of current bar (like backtest) instead of just lastPrice
    """
    global open_positions, binance_client
    
    if not binance_client or not open_positions:
        return
    
    try:
        # BACKTEST CLONE: Get current bar HIGH/LOW (not just lastPrice)
        # This matches backtest behavior exactly (checks if HIGH touched TP or LOW touched SL)
        klines = binance_client.get_klines(symbol='BTCUSDT', interval='5m', limit=2)
        if not klines or len(klines) < 1:
            return
        
        # Get the latest completed bar (last one in array)
        latest_bar = klines[-1]
        current_high = float(latest_bar['high'])
        current_low = float(latest_bar['low'])
        current_open = float(latest_bar['open'])
        current_close = float(latest_bar['close'])
        bar_close_time = int(latest_bar['close_time'])
        
        positions_to_close = []
        
        for position_id, position in open_positions.items():
            entry_price = position['entry_price']
            side = position['side']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            should_close = False
            close_reason = ""
            exit_price = None
            
            if side == 'BUY':
                # Long position - BACKTEST CLONE: Check if LOW touched stop-loss OR HIGH touched take-profit
                # Priority: Stop-loss first (risk management) - same as backtest
                if current_low <= stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                    exit_price = stop_loss  # Execute at stop-loss price (same as backtest)
                elif current_high >= take_profit:
                    should_close = True
                    close_reason = "take_profit"
                    exit_price = take_profit  # Execute at take-profit price (same as backtest)
            else:
                # Short position (if implemented)
                # BACKTEST CLONE: Check if HIGH touched stop-loss OR LOW touched take-profit
                if current_high >= stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                    exit_price = stop_loss
                elif current_low <= take_profit:
                    should_close = True
                    close_reason = "take_profit"
                    exit_price = take_profit
            
            if should_close:
                positions_to_close.append((position_id, close_reason, exit_price))
        
        # Close positions that hit stop-loss or take-profit
        for position_id, reason, exit_price in positions_to_close:
            close_position_sync(position_id, reason)
            
    except Exception as e:
        logger.error(f"❌ Position management error: {e}")

def close_position_sync(position_id, reason):
    """Close a specific position (sync version) - Converts ALL BTC to USDT"""
    global open_positions, binance_client, position_closing_in_progress
    
    if position_id not in open_positions:
        logger.debug(f"⏸️ Position {position_id} not found in tracked positions - already closed?")
        return
    
    # Prevent multiple simultaneous close operations
    if position_closing_in_progress:
        logger.warning(f"⚠️ Position closing already in progress - skipping {position_id}")
        return
    
    position = open_positions[position_id]
    position_closing_in_progress = True
    
    try:
        logger.info(f"🔒 Closing position: {position_id} - Reason: {reason} - Entry: ${position['entry_price']:.2f}")
        # Get current account balance - check BOTH free and locked BTC
        account = binance_client.get_account()
        btc_balance_free = 0.0
        btc_balance_locked = 0.0
        btc_total = 0.0
        
        for balance in account['balances']:
            if balance['asset'] == 'BTC':
                btc_balance_free = float(balance['free'])
                btc_balance_locked = float(balance['locked'])
                btc_total = btc_balance_free + btc_balance_locked
                break
        
        # CRITICAL FIX: Sell ALL BTC (free + locked) to ensure 0 BTC balance
        if btc_total > 0:
            # Wait a moment for any pending orders to settle
            time.sleep(1)
            
            # Re-check balance after waiting (in case locked became free)
            account = binance_client.get_account()
            btc_balance_free = 0.0
            btc_balance_locked = 0.0
            for balance in account['balances']:
                if balance['asset'] == 'BTC':
                    btc_balance_free = float(balance['free'])
                    btc_balance_locked = float(balance['locked'])
                    break
            
            # Use only FREE balance for immediate sell (locked will be sold when it becomes free)
            if btc_balance_free > 0:
                # Execute SELL order - convert ALL free BTC to USDT
                # Binance minimum is typically 0.00001 BTC
                if btc_balance_free >= 0.00001:
                    logger.info(f"📤 Executing SINGLE market SELL order: {btc_balance_free:.8f} BTC (position: {position_id})")
                    order = binance_client.order_market_sell(
                        symbol='BTCUSDT',
                        quantity=f"{btc_balance_free:.8f}"  # Use 8 decimals for precision
                    )
                    
                    logger.info(f"✅ Position closed: {position_id} - {reason} - Sold {btc_balance_free:.8f} BTC - Order ID: {order.get('orderId', 'N/A')}")
                else:
                    logger.warning(f"⚠️ BTC balance too small to sell: {btc_balance_free:.8f} BTC (minimum 0.00001)")
            
            # If there's locked BTC, log it but it will be sold later when it becomes free
            if btc_balance_locked > 0:
                logger.warning(f"⚠️ {btc_balance_locked:.8f} BTC is locked - will be converted when unlocked")
            
            # Verify BTC balance is 0 after sell (with small tolerance for rounding)
            time.sleep(1)  # Wait for order to settle
            account = binance_client.get_account()
            btc_after_sell = 0.0
            for balance in account['balances']:
                if balance['asset'] == 'BTC':
                    btc_after_sell = float(balance['free'])
                    break
            
            if btc_after_sell > 0.00001:  # If still > minimum, try selling again
                logger.warning(f"⚠️ Remaining BTC balance detected: {btc_after_sell:.8f} BTC - Attempting to sell...")
                try:
                    order_retry = binance_client.order_market_sell(
                        symbol='BTCUSDT',
                        quantity=f"{btc_after_sell:.8f}"
                    )
                    logger.info(f"✅ Converted remaining BTC: {btc_after_sell:.8f} BTC")
                except Exception as retry_error:
                    logger.error(f"❌ Failed to convert remaining BTC: {retry_error}")
            else:
                logger.info(f"✅ BTC balance verified: {btc_after_sell:.8f} BTC (near zero)")
            
        # CRITICAL: Remove from tracking AFTER selling to prevent re-triggering
        # Wait a bit longer to ensure order is fully processed
        time.sleep(2)
        if position_id in open_positions:
            del open_positions[position_id]
            logger.info(f"🗑️ Removed position {position_id} from tracking")
        else:
            logger.warning(f"⚠️ Position {position_id} was already removed from tracking")
            
    except Exception as e:
        logger.error(f"❌ Error closing position {position_id}: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
    finally:
        # Always release the lock
        position_closing_in_progress = False
        logger.debug(f"🔓 Released position closing lock")

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
            
            # CRITICAL FIX #0: Check for leftover BTC balance and convert to USDT
            # This ensures we always have 0 BTC when no positions are tracked
            # BUT: Skip if position closing is in progress to avoid race conditions
            if not open_positions and not position_closing_in_progress:
                try:
                    account = binance_client.get_account()
                    btc_balance = 0.0
                    for balance in account['balances']:
                        if balance['asset'] == 'BTC':
                            btc_balance = float(balance['free'])
                            break
                    
                    # If there's BTC but no tracked positions, convert it to USDT
                    # Add delay to ensure previous order is fully processed
                    if btc_balance > 0.00001:  # Above minimum tradeable amount
                        logger.warning(f"⚠️ Leftover BTC balance detected: {btc_balance:.8f} BTC (no tracked positions) - Waiting 3s then converting to USDT...")
                        time.sleep(3)  # Wait to ensure previous orders are settled
                        try:
                            # Re-check balance after waiting (might have changed)
                            account = binance_client.get_account()
                            btc_balance = 0.0
                            for balance in account['balances']:
                                if balance['asset'] == 'BTC':
                                    btc_balance = float(balance['free'])
                                    break
                            
                            if btc_balance > 0.00001:  # Still have BTC after waiting
                                ticker = binance_client.get_ticker(symbol='BTCUSDT')
                                current_price = float(ticker['lastPrice'])
                                
                                # Convert all BTC to USDT - SINGLE order
                                logger.info(f"📤 Executing SINGLE cleanup SELL order: {btc_balance:.8f} BTC")
                                order = binance_client.order_market_sell(
                                    symbol='BTCUSDT',
                                    quantity=f"{btc_balance:.8f}"
                                )
                                logger.info(f"✅ Cleanup order executed - Order ID: {order.get('orderId', 'N/A')}")
                            
                            # Verify conversion
                            time.sleep(2)  # Longer wait for order to settle
                            account = binance_client.get_account()
                            btc_after = 0.0
                            for balance in account['balances']:
                                if balance['asset'] == 'BTC':
                                    btc_after = float(balance['free'])
                                    break
                            
                            if btc_after < 0.00001:
                                logger.info(f"✅ Successfully converted {btc_balance:.8f} BTC to USDT")
                            else:
                                logger.warning(f"⚠️ Still {btc_after:.8f} BTC remaining after conversion")
                        except Exception as convert_error:
                            logger.error(f"❌ Failed to convert leftover BTC: {convert_error}")
                            # Continue anyway to avoid blocking the loop
                except Exception as check_error:
                    logger.error(f"❌ Error checking BTC balance: {check_error}")
            
            # BACKTEST CLONE: Apply step_size (only evaluate 1 bar out of 6)
            # This matches backtest exactly: step_size = 6 means evaluate only every 6th bar
            try:
                # Get recent bars to check step_size logic
                klines = binance_client.get_klines(symbol='BTCUSDT', interval='5m', limit=step_size * 2)
                if not klines or len(klines) < step_size:
                    time.sleep(10)
                    continue
                
                # Get the latest completed bar (use the one before last to ensure it's closed)
                # In live trading, the last bar might still be forming
                latest_bar_index = len(klines) - 2 if len(klines) >= 2 else len(klines) - 1
                latest_bar_time = int(klines[latest_bar_index]['close_time'])
                
                # BACKTEST CLONE: Check if we should evaluate this bar (step_size logic)
                should_evaluate = False
                if last_evaluated_bar_time is None:
                    # First evaluation - evaluate current bar
                    should_evaluate = True
                else:
                    # Find how many bars have passed since last evaluation
                    bars_since_last = 0
                    found_last = False
                    
                    # Look for last evaluated bar in recent klines
                    for i in range(latest_bar_index, -1, -1):
                        if int(klines[i]['close_time']) == last_evaluated_bar_time:
                            bars_since_last = latest_bar_index - i
                            found_last = True
                            break
                    
                    # If last evaluated bar not found, check if enough time has passed
                    if not found_last:
                        # Calculate time difference (approximate)
                        time_diff_ms = latest_bar_time - last_evaluated_bar_time
                        # 5min bars = 300000 ms each
                        estimated_bars = time_diff_ms // 300000
                        if estimated_bars >= step_size:
                            bars_since_last = step_size
                    
                    # Evaluate only if step_size bars have passed (same as backtest)
                    if bars_since_last >= step_size:
                        should_evaluate = True
                        logger.debug(f"✅ Step size: {bars_since_last} bars since last evaluation (>= {step_size})")
                
                if not should_evaluate:
                    logger.debug(f"⏸️ Step size: Skipping evaluation - only {bars_since_last if 'bars_since_last' in locals() else 0} bars since last (need {step_size})")
                    # Still check positions but skip signal evaluation
                    # Position management happens every loop iteration
                    time.sleep(30)
                    continue
                
                # Update last evaluated bar time (BACKTEST CLONE)
                last_evaluated_bar_time = latest_bar_time
                logger.info(f"📊 BACKTEST CLONE: Evaluating bar {latest_bar_time} (step_size={step_size}, 1 out of {step_size} bars evaluated)")
                
            except Exception as step_error:
                logger.warning(f"⚠️ Error in step_size check: {step_error} - Skipping evaluation this cycle")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                time.sleep(30)
                continue  # Skip this evaluation cycle if step check fails
            
            # Get ML signal
            signal_data = get_ml_signal()
            
            # Check if signal meets criteria for automatic execution
            # Use same threshold as backtest for consistency (42% = 68% win rate)
            if signal_data['confidence'] >= 0.42 and signal_data['signal'] != 'HOLD':
                # CRITICAL FIX #1: Check if we already have open positions (prevent multiple simultaneous trades)
                if signal_data['signal'] == 'BUY' and open_positions:
                    logger.debug(f"⏸️ BUY signal ignored: {len(open_positions)} open positions already exist")
                    time.sleep(10)
                    continue
                
                # BACKTEST CLONE: Apply EXACT same filters as backtest
                # This ensures live trading matches backtest behavior (1 trade per 2 days vs 1 trade per 2-4 hours)
                if signal_data['signal'] == 'BUY':
                    # BACKTEST CLONE: Apply confidence threshold FIRST (same as backtest line 1288-1291)
                    confidence = signal_data['confidence']
                    if confidence < ml_system.long_confidence_threshold:
                        logger.debug(f"⏸️ BUY signal rejected: confidence {confidence:.3f} < {ml_system.long_confidence_threshold} (backtest filter)")
                        time.sleep(10)
                        continue
                    
                    # BACKTEST CLONE: Apply _should_allow_long_signal filter (same as backtest line 1293-1299)
                    try:
                        # Need to get the features dataframe to check filters
                        required_klines = max(ml_system.lookback_period + 100, 300)
                        klines = binance_client.get_klines(symbol='BTCUSDT', interval='5m', limit=required_klines)
                        df = pd.DataFrame(klines, columns=[
                            'open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base','taker_quote','ignore'
                        ])
                        df = df[['open_time','open','high','low','close','volume']].copy()
                        df['kline_timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                        for col in ['open','high','low','close','volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        df = df.dropna()
                        df_renamed = df.copy()
                        df_renamed['timestamp'] = df_renamed['kline_timestamp']
                        drop_cols = [c for c in ['kline_timestamp', 'open_time'] if c in df_renamed.columns]
                        if drop_cols:
                            df_renamed = df_renamed.drop(columns=drop_cols)
                        
                        # Get current index (last row) - BACKTEST CLONE
                        current_idx = len(df_renamed) - 1
                        prediction = 1  # BUY signal
                        
                        # BACKTEST CLONE: Apply the exact same filter as backtest (line 1293)
                        allowed, reason = ml_system._should_allow_long_signal(df_renamed, current_idx, confidence, prediction)
                        
                        if not allowed:
                            logger.debug(f"⏸️ BUY signal filtered out (backtest filter): {reason}")
                            time.sleep(10)
                            continue
                        else:
                            logger.debug(f"✅ BUY signal passed all backtest filters: {reason}")
                    except Exception as filter_error:
                        logger.warning(f"⚠️ Could not apply backtest filters: {filter_error} - Rejecting trade for safety")
                        time.sleep(10)
                        continue  # Reject trade if filter check fails (safer)
                
                
                # CRITICAL FIX #2: DISABLE SELL signals (same as backtest - SHORT trades have 1-3% win rate)
                # NEVER sell BTC if there are tracked positions - positions must hit TP/SL first
                if signal_data['signal'] == 'SELL':
                    # CRITICAL: Check if there's a tracked position FIRST
                    # If tracked position exists, BTC belongs to that position - don't sell it!
                    if open_positions:
                        logger.debug(f"⏸️ SELL signal ignored: {len(open_positions)} tracked positions exist - BTC must stay until TP/SL")
                        time.sleep(10)
                        continue
                    
                    # Also check if position closing is in progress
                    if position_closing_in_progress:
                        logger.debug(f"⏸️ SELL signal ignored: Position closing in progress - wait for completion")
                        time.sleep(10)
                        continue
                    
                    # Only convert BTC if NO tracked positions exist (leftover BTC cleanup only)
                    try:
                        account = binance_client.get_account()
                        btc_balance = 0.0
                        for balance in account['balances']:
                            if balance['asset'] == 'BTC':
                                btc_balance = float(balance['free'])
                                break
                        
                        if btc_balance > 0.00001:
                            logger.warning(f"⚠️ SELL signal + leftover BTC (no tracked positions): {btc_balance:.8f} BTC - Waiting 3s then converting...")
                            time.sleep(3)  # Wait to ensure previous orders are settled
                            
                            # Re-check balance
                            account = binance_client.get_account()
                            btc_balance = 0.0
                            for balance in account['balances']:
                                if balance['asset'] == 'BTC':
                                    btc_balance = float(balance['free'])
                                    break
                            
                            if btc_balance > 0.00001:  # Still have BTC
                                try:
                                    logger.info(f"📤 Executing SINGLE SELL signal cleanup order: {btc_balance:.8f} BTC")
                                    order = binance_client.order_market_sell(
                                        symbol='BTCUSDT',
                                        quantity=f"{btc_balance:.8f}"
                                    )
                                    logger.info(f"✅ Converted leftover BTC: {btc_balance:.8f} BTC to USDT - Order ID: {order.get('orderId', 'N/A')}")
                                except Exception as convert_error:
                                    logger.error(f"❌ Failed to convert BTC on SELL signal: {convert_error}")
                        else:
                            logger.debug(f"⏸️ SELL signal ignored: SHORT trades disabled (no BTC to convert)")
                    except Exception as check_error:
                        logger.debug(f"⏸️ SELL signal ignored: SHORT trades disabled (error checking balance: {check_error})")
                    
                    time.sleep(10)
                    continue
                
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
                        # Use AVAILABLE capital for position sizing (allows compounding growth)
                        # position_size from ML (35-70%) applied to current available balance
                        usdt_to_spend = usdt_balance * signal_data['position_size']
                        
                        # Ensure we don't spend more than available (leave 1% for fees)
                        usdt_to_spend = min(usdt_to_spend, usdt_balance * 0.99)
                        
                        logger.info(f"💰 Position sizing: usdt_available={usdt_balance:.2f}, position_size={signal_data['position_size']:.2%}, final_amount={usdt_to_spend:.2f}")
                        
                        # Check minimum order size (Binance minimum is typically 10 USDT)
                        if usdt_to_spend < 10:
                            logger.warning(f"⚠️ Order size too small: {usdt_to_spend:.2f} USDT (minimum 10 USDT)")
                            time.sleep(10)
                            continue
                        
                        # BACKTEST CLONE: Execute at bar close price (not current market price)
                        # Get the latest completed bar close price (matches backtest execution)
                        klines_bar = binance_client.get_klines(symbol='BTCUSDT', interval='5m', limit=1)
                        if klines_bar and len(klines_bar) > 0:
                            # Use close price of the bar (same as backtest)
                            execution_price = float(klines_bar[0]['close'])
                            logger.debug(f"📊 Using bar close price for execution: {execution_price:.2f} (backtest clone)")
                        else:
                            # Fallback to current price if bar not available
                            execution_price = current_price
                            logger.warning(f"⚠️ Could not get bar close price, using current price: {current_price:.2f}")
                        
                        # Calculate BTC quantity from USDT amount
                        btc_quantity = usdt_to_spend / execution_price
                        
                        # Execute BUY order
                        order = binance_client.order_market_buy(
                            symbol='BTCUSDT',
                            quoteOrderQty=f"{usdt_to_spend:.2f}"  # Use quoteOrderQty (USDT amount) instead of quantity
                        )
                        logger.info(f"✅ Auto-executed BUY order: {order['orderId']} - {usdt_to_spend:.2f} USDT (~{btc_quantity:.6f} BTC) @ {execution_price:.2f}")
                        
                        # BACKTEST CLONE: Track the position for stop-loss/take-profit management
                        position_id = f"pos_{int(time.time())}"
                        # Use long_stop_loss_pct from TargetedLongLossFixSystem (0.6%)
                        stop_loss_pct = getattr(ml_system, 'long_stop_loss_pct', 0.006)
                        # BACKTEST CLONE: Use execution price (bar close) for SL/TP calculation
                        stop_loss_price = execution_price * (1 - stop_loss_pct)
                        # Use dynamic take-profit from signal (already calculated)
                        take_profit_price = signal_data.get('take_profit', execution_price * 1.015)  # Dynamic TP from signal
                        
                        open_positions[position_id] = {
                            'side': 'BUY',
                            'entry_price': execution_price,  # BACKTEST CLONE: Use bar close price, not current price
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'entry_time': datetime.now(),
                            'order_id': order['orderId'],
                            'entry_confidence': confidence  # Store confidence for logging
                        }
                        
                        logger.info(f"📊 Position tracked: {position_id} - Entry: {execution_price:.2f} (bar close), SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}, Confidence: {confidence:.1%}")
                    
                except Exception as trade_error:
                    logger.error(f"❌ Auto-trade execution failed: {trade_error}")
            
            else:
                logger.debug(f"🔍 Auto-trading: Signal {signal_data['signal']} with {signal_data['confidence']:.1%} confidence - No action")
            
            # BACKTEST CLONE: Wait longer to match backtest frequency (step_size = 6 bars = 30 minutes)
            # Since we evaluate 1 bar out of 6, we wait for next bar (5 minutes = 300 seconds)
            # But check more frequently for position management (every 30 seconds)
            time.sleep(30)  # Check every 30 seconds for position management, but only evaluate signals at step_size intervals
            
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
        # Need at least lookback_period + some buffer for feature calculation
        required_klines = max(ml_system.lookback_period + 100, 300)
        klines = binance_client.get_klines(symbol='BTCUSDT', interval='5m', limit=required_klines)
        if not klines or len(klines) < ml_system.lookback_period:
            raise RuntimeError(f"Not enough klines: got {len(klines) if klines else 0}, need at least {ml_system.lookback_period}")
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
        # Use TargetedLongLossFixSystem's create_targeted_features method
        df_features = ml_system.create_targeted_features(df_renamed)
        if df_features is None or len(df_features) == 0:
            raise RuntimeError("Feature creation returned empty dataframe")
        
        # Get the latest row
        latest_row = df_features.iloc[-1]
        
        # CRITICAL: Use selected_features in the EXACT order they were trained
        # This ensures compatibility with the scaler and models
        if hasattr(ml_system, 'selected_features') and ml_system.selected_features:
            # Use features in the exact order they were saved during training
            # This matches the order used during model training
            feature_list = ml_system.selected_features
            logger.debug(f"Using {len(feature_list)} features from selected_features.json")
            
            # Verify all features exist in the dataframe
            missing_features = [f for f in feature_list if f not in df_features.columns]
            if missing_features:
                logger.error(f"❌ Missing features: {missing_features[:10]}...")
                raise RuntimeError(f"Missing {len(missing_features)} required features in dataframe")
            
            # Extract features in the exact training order
            X_row = latest_row[feature_list].values.reshape(1, -1)
        else:
            # Fallback: use all features except metadata columns
            feature_columns = [c for c in df_features.columns if c not in ['timestamp','target','open','high','low','close','volume']]
            logger.warning(f"⚠️ No selected_features found, using all {len(feature_columns)} features")
            X_row = latest_row[feature_columns].values.reshape(1, -1)
        
        # Scale features - scaler expects exact same features in same order as training
        if not hasattr(ml_system, 'scaler') or ml_system.scaler is None:
            raise RuntimeError("Scaler not loaded. Train and save models first or provide scaler.pkl")
        
        # Verify feature count matches scaler expectation
        scaler_n_features = ml_system.scaler.n_features_in_ if hasattr(ml_system.scaler, 'n_features_in_') else None
        if scaler_n_features and X_row.shape[1] != scaler_n_features:
            error_msg = f"Feature count mismatch: Expected {scaler_n_features} features (from scaler), got {X_row.shape[1]}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        X_scaled = ml_system.scaler.transform(X_row)
        # Predict via ensemble (same logic as algorithm's _ensemble_predict)
        if not hasattr(ml_system, 'models') or not ml_system.models:
            raise RuntimeError("Models not loaded. Train and save models first or provide model pkl files")
        # Collect all probabilities first (same as algorithm)
        all_probabilities = []
        for name, model in ml_system.models.items():
            try:
                logger.debug(f"🔄 Calling predict_proba on {name} with shape {X_scaled.shape}")
                prob = model.predict_proba(X_scaled)
                all_probabilities.append(prob)
            except Exception as pred_error:
                logger.error(f"❌ Error in predict_proba for {name}: {pred_error}")
                import traceback
                logger.error(f"❌ Traceback: {traceback.format_exc()}")
                raise
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
        # Calculate position size using TargetedLongLossFixSystem method
        position_size = float(ml_system._calculate_position_size(confidence))
        # No leverage in TargetedLongLossFixSystem, use 1.0
        leverage = 1.0
        stop_loss = current_price * (1 - ml_system.long_stop_loss_pct)
        # Use dynamic take-profit based on confidence
        dynamic_tp = ml_system._calculate_dynamic_take_profit(confidence, 'LONG')
        take_profit = current_price * (1 + dynamic_tp)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': f'Ensemble ML prediction with {confidence:.1%} confidence',
            'leverage': leverage,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': current_price,
            'algorithm': 'Targeted Long Loss Fix System (0.6% SL)',
            'models_loaded': True,
            'trading_halted': getattr(ml_system, 'trading_halted', False)
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
    """Convert BTC to USDT (useful for getting trading funds)
    
    Note: If you want to convert ALL BTC to USDT, set percent=1.0 or use /api/convert-all-btc-to-usdt
    """
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
            quantity=f"{btc_to_sell:.8f}"  # Use 8 decimals for precision
        )
        
        # Verify conversion
        time.sleep(1)
        account = binance_client.get_account()
        btc_after = 0.0
        for balance in account['balances']:
            if balance['asset'] == 'BTC':
                btc_after = float(balance['free'])
                break
        
        return {
            "status": "success", 
            "sold_btc": btc_to_sell, 
            "approx_usdt": btc_to_sell * current_price, 
            "order_id": order['orderId'],
            "remaining_btc": btc_after
        }
    except Exception as e:
        logger.error(f"❌ Error converting BTC to USDT: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/convert-all-btc-to-usdt")
async def convert_all_btc_to_usdt():
    """Convert ALL BTC balance to USDT - ensures 0 BTC balance
    
    This is useful for:
    - Cleaning up leftover BTC after trades
    - Ensuring no BTC balance remains after position closes
    - Manual cleanup when needed
    """
    if not binance_client:
        raise HTTPException(status_code=400, detail="Binance API not configured")

    try:
        account = binance_client.get_account()
        btc_balance_free = 0.0
        btc_balance_locked = 0.0
        
        for balance in account['balances']:
            if balance['asset'] == 'BTC':
                btc_balance_free = float(balance['free'])
                btc_balance_locked = float(balance['locked'])
                break

        if btc_balance_free <= 0.00001:
            return {
                "status": "success",
                "message": "No BTC to convert (balance already near zero)",
                "btc_balance": btc_balance_free,
                "btc_locked": btc_balance_locked
            }

        ticker = binance_client.get_ticker(symbol='BTCUSDT')
        current_price = float(ticker['lastPrice'])
        
        # Convert ALL free BTC to USDT
        order = binance_client.order_market_sell(
            symbol='BTCUSDT',
            quantity=f"{btc_balance_free:.8f}"
        )
        
        # Verify conversion
        time.sleep(1)
        account = binance_client.get_account()
        btc_after = 0.0
        for balance in account['balances']:
            if balance['asset'] == 'BTC':
                btc_after = float(balance['free'])
                break
        
        return {
            "status": "success",
            "message": "All BTC converted to USDT",
            "sold_btc": btc_balance_free,
            "approx_usdt": btc_balance_free * current_price,
            "order_id": order['orderId'],
            "remaining_btc": btc_after,
            "btc_locked": btc_balance_locked,
            "note": "Locked BTC will be converted when it becomes free"
        }
    except Exception as e:
        logger.error(f"❌ Error converting all BTC to USDT: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
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