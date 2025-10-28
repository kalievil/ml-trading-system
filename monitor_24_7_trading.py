#!/usr/bin/env python3

"""
24/7 ML Trading Monitor
Monitors the performance of your automated ML trading algorithm
"""

import requests
import time
import json
from datetime import datetime

def get_trading_status():
    """Get current trading status"""
    try:
        response = requests.get("http://localhost:8000/api/auto-trading-status")
        return response.json()
    except:
        return None

def get_performance():
    """Get trading performance metrics"""
    try:
        response = requests.get("http://localhost:8000/api/trading-performance")
        return response.json()
    except:
        return None

def get_account_info():
    """Get account information"""
    try:
        response = requests.get("http://localhost:8000/api/account-info")
        return response.json()
    except:
        return None

def monitor_trading():
    """Monitor 24/7 trading performance"""
    print("🤖 24/7 ML Trading Monitor Started")
    print("=" * 50)
    
    while True:
        try:
            # Get current status
            status = get_trading_status()
            performance = get_performance()
            account = get_account_info()
            
            if status and performance and account:
                print(f"\n📊 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 30)
                print(f"🤖 Auto Trading: {'✅ ENABLED' if status['enabled'] else '❌ DISABLED'}")
                print(f"🎯 ML Engine: {'✅ READY' if status['trading_engine_ready'] else '❌ NOT READY'}")
                print(f"🔗 Binance: {'✅ CONNECTED' if status['binance_connected'] else '❌ DISCONNECTED'}")
                print(f"⚡ Check Interval: 10 seconds")
                print(f"🎯 Trading Mode: Unrestricted ML signals")
                print(f"💰 Portfolio Value: ${account.get('total_wallet_balance', 0):,.2f}")
                print(f"📈 Total Return: {performance.get('total_return_percent', 0):.2f}%")
                print(f"🎯 Win Rate: {performance.get('win_rate_percent', 0):.1f}%")
                print(f"📊 Total Trades: {performance.get('total_trades', 0)}")
                print(f"💵 Total P&L: ${performance.get('total_pnl', 0):.2f}")
                
                if performance.get('total_trades', 0) > 0:
                    print(f"🏆 Profitable Trades: {performance.get('profitable_trades', 0)}")
                    print(f"📉 Losing Trades: {performance.get('losing_trades', 0)}")
                    print(f"💎 Avg Win: ${performance.get('average_win', 0):.2f}")
                    print(f"📉 Avg Loss: ${performance.get('average_loss', 0):.2f}")
                
                print("-" * 30)
            else:
                print(f"❌ {datetime.now().strftime('%H:%M:%S')} - API Connection Error")
            
            # Wait 60 seconds before next check
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_trading()
