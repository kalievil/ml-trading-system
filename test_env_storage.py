#!/usr/bin/env python3

"""
Test script to demonstrate environment variable storage
"""

import os
import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_environment_storage():
    """Test the environment variable storage functionality"""
    
    print("🧪 Testing Environment Variable Storage")
    print("=" * 50)
    
    # Test 1: Check initial health
    print("\n1️⃣ Checking initial health...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health")
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   Models Loaded: {health['models_loaded']}")
        print(f"   Binance Configured: {health['binance_configured']}")
        print(f"   Credentials from ENV: {health['credentials_from_env']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Configure with test credentials
    print("\n2️⃣ Configuring with test credentials...")
    test_credentials = {
        "api_key": "test_api_key_12345",
        "api_secret": "test_secret_key_67890"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/configure-binance",
            json=test_credentials,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {result['message']}")
            print(f"   📦 Persistent Storage: {result['persistent_storage']}")
        else:
            print(f"   ❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Check health after configuration
    print("\n3️⃣ Checking health after configuration...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health")
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   Binance Configured: {health['binance_configured']}")
        print(f"   Credentials from ENV: {health['credentials_from_env']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Show environment variables
    print("\n4️⃣ Environment Variables:")
    api_key = os.environ.get('BINANCE_API_KEY', 'Not set')
    api_secret = os.environ.get('BINANCE_API_SECRET', 'Not set')
    print(f"   BINANCE_API_KEY: {api_key[:10]}..." if api_key != 'Not set' else f"   BINANCE_API_KEY: {api_key}")
    print(f"   BINANCE_API_SECRET: {api_secret[:10]}..." if api_secret != 'Not set' else f"   BINANCE_API_SECRET: {api_secret}")
    
    print("\n🎉 Environment variable storage test completed!")
    print("\n💡 For Render deployment:")
    print("   1. Set BINANCE_API_KEY and BINANCE_API_SECRET in Render dashboard")
    print("   2. Server will auto-load credentials on startup")
    print("   3. Credentials persist through server restarts")

if __name__ == "__main__":
    test_environment_storage()
