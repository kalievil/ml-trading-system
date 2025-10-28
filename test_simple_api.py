#!/usr/bin/env python3

"""
Test script for simple Binance API verification
"""

import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_binance_api():
    """Test the simple Binance API endpoints"""
    
    print("🧪 Testing Simple Binance API Integration")
    print("=" * 50)
    
    # Test 1: Check API health
    print("\n1️⃣ Checking API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health")
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   Models Loaded: {health['models_loaded']}")
        print(f"   Binance Configured: {health['binance_configured']}")
        print(f"   Credentials from File: {health['credentials_from_file']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Generate test signal
    print("\n2️⃣ Generating test signal...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/test-signal")
        signal = response.json()
        print(f"   Symbol: {signal['symbol']}")
        print(f"   Side: {signal['side']}")
        print(f"   Entry Price: ${signal['entry_price']:.2f}")
        print(f"   Confidence: {signal['confidence']:.2f}")
        print(f"   Test Mode: {signal['test_mode']}")
        print(f"   Message: {signal['message']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Execute test trade
    print("\n3️⃣ Executing test trade...")
    try:
        response = requests.post(f"{API_BASE_URL}/api/test-execute")
        trade = response.json()
        print(f"   Success: {trade['success']}")
        print(f"   Test Mode: {trade['test_mode']}")
        print(f"   Order ID: {trade['order_id']}")
        print(f"   Symbol: {trade['symbol']}")
        print(f"   Side: {trade['side']}")
        print(f"   Quantity: {trade['quantity']}")
        print(f"   Price: ${trade['price']}")
        print(f"   Status: {trade['status']}")
        print(f"   Message: {trade['message']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🎉 Simple API testing completed!")
    print("\n💡 Next steps:")
    print("   1. Check your Binance Testnet account for the test trade")
    print("   2. If successful, the API is working correctly")
    print("   3. You can now restore the ML algorithm")

if __name__ == "__main__":
    test_binance_api()
