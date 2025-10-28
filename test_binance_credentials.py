#!/usr/bin/env python3

"""
Test Binance credentials manually
"""

import json
from binance.client import Client
from pathlib import Path

def test_binance_credentials():
    """Test the Binance credentials from file"""
    
    credentials_file = Path("binance_credentials.json")
    
    if not credentials_file.exists():
        print("❌ Credentials file not found")
        return
    
    try:
        with open(credentials_file, 'r', encoding='utf-8') as f:
            credentials = json.load(f)
        
        api_key = credentials.get('api_key')
        api_secret = credentials.get('api_secret')
        
        print(f"🔑 API Key: {api_key[:10]}..." if api_key else "❌ No API key")
        print(f"🔑 API Secret: {api_secret[:10]}..." if api_secret else "❌ No API secret")
        
        if not api_key or not api_secret:
            print("❌ Missing credentials")
            return
        
        print("🔧 Initializing Binance client...")
        client = Client(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True
        )
        
        print("🔧 Testing connection...")
        account_info = client.get_account()
        
        print("✅ Connection successful!")
        print(f"📊 Account type: {account_info.get('accountType', 'Unknown')}")
        print(f"📊 Can trade: {account_info.get('canTrade', False)}")
        
        # Test getting current price
        print("🔧 Testing market data...")
        ticker = client.get_symbol_ticker(symbol="BTCUSDT")
        print(f"📈 BTC Price: ${ticker['price']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"❌ Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_binance_credentials()
