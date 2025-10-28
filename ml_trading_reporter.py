#!/usr/bin/env python3

"""
ML Trading Report Generator
Creates HTML reports from JSON data files for monitoring ML trading performance
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import requests
from typing import Dict, List, Any

class MLTradingReporter:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.reports_dir = Path("trading_reports")
        self.data_dir = Path("trading_data")
        
        # Create directories
        self.reports_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        print("📊 ML Trading Reporter initialized")
        print(f"📁 Reports directory: {self.reports_dir}")
        print(f"📁 Data directory: {self.data_dir}")
    
    def fetch_api_data(self) -> Dict[str, Any]:
        """Fetch data from the ML trading API"""
        try:
            # Health check
            health_response = requests.get(f"{self.api_base_url}/api/health", timeout=5)
            health = health_response.json() if health_response.status_code == 200 else {}
            
            # Account info
            account_response = requests.get(f"{self.api_base_url}/api/account-info", timeout=5)
            account_info = account_response.json() if account_response.status_code == 200 else {}
            
            # Positions
            positions_response = requests.get(f"{self.api_base_url}/api/positions", timeout=5)
            positions = positions_response.json() if positions_response.status_code == 200 else []
            
            # Trade history
            history_response = requests.get(f"{self.api_base_url}/api/trade-history", timeout=5)
            trade_history = history_response.json() if history_response.status_code == 200 else []
            
            # Trading performance
            performance_response = requests.get(f"{self.api_base_url}/api/trading-performance", timeout=5)
            performance = performance_response.json() if performance_response.status_code == 200 else {}
            
            # ML signal
            signal_response = requests.get(f"{self.api_base_url}/api/ml-signal", timeout=5)
            ml_signal = signal_response.json() if signal_response.status_code == 200 else {}
            
            # Auto trading status
            auto_status_response = requests.get(f"{self.api_base_url}/api/auto-trading-status", timeout=5)
            auto_status = auto_status_response.json() if auto_status_response.status_code == 200 else {}
            
            return {
                "timestamp": datetime.now().isoformat(),
                "health": health,
                "account_info": account_info,
                "positions": positions,
                "trade_history": trade_history,
                "performance": performance,
                "ml_signal": ml_signal,
                "auto_trading_status": auto_status
            }
            
        except Exception as e:
            print(f"❌ Error fetching API data: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "health": {},
                "account_info": {},
                "positions": [],
                "trade_history": [],
                "performance": {},
                "ml_signal": {},
                "auto_trading_status": {}
            }
    
    def save_data_to_json(self, data: Dict[str, Any]):
        """Save data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_data_{timestamp}.json"
        filepath = self.data_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Data saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def load_latest_data(self) -> Dict[str, Any]:
        """Load the most recent data file"""
        try:
            data_files = list(self.data_dir.glob("trading_data_*.json"))
            if not data_files:
                return {}
            
            # Get the most recent file
            latest_file = max(data_files, key=os.path.getctime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return {}
    
    def generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML report from data"""
        
        # Extract data
        health = data.get("health", {})
        account_info = data.get("account_info", {})
        positions = data.get("positions", [])
        trade_history = data.get("trade_history", [])
        performance = data.get("performance", {})
        ml_signal = data.get("ml_signal", {})
        auto_status = data.get("auto_trading_status", {})
        timestamp = data.get("timestamp", "")
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = timestamp
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Trading Report - {formatted_time}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .status-bar {{
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        .status-item {{
            text-align: center;
        }}
        .status-item h3 {{
            margin: 0 0 5px 0;
            color: #495057;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .status-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #28a745;
        }}
        .status-value.error {{
            color: #dc3545;
        }}
        .status-value.warning {{
            color: #ffc107;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #495057;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .metric-card h4 {{
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-value.positive {{
            color: #28a745;
        }}
        .metric-value.negative {{
            color: #dc3545;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        .signal-buy {{
            color: #28a745;
            font-weight: bold;
        }}
        .signal-sell {{
            color: #dc3545;
            font-weight: bold;
        }}
        .signal-hold {{
            color: #6c757d;
            font-weight: bold;
        }}
        .auto-trading-enabled {{
            color: #28a745;
            font-weight: bold;
        }}
        .auto-trading-disabled {{
            color: #dc3545;
            font-weight: bold;
        }}
        .refresh-info {{
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 ML Trading Report</h1>
            <p>Optimized High-Return System - Binance Testnet</p>
            <p>Last Updated: {formatted_time}</p>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <h3>API Status</h3>
                <div class="status-value {'error' if not health.get('status') == 'healthy' else ''}">
                    {health.get('status', 'Unknown').upper()}
                </div>
            </div>
            <div class="status-item">
                <h3>ML System</h3>
                <div class="status-value {'error' if not health.get('ml_system_ready') else ''}">
                    {'READY' if health.get('ml_system_ready') else 'NOT READY'}
                </div>
            </div>
            <div class="status-item">
                <h3>Binance</h3>
                <div class="status-value {'error' if not health.get('binance_configured') else ''}">
                    {'CONNECTED' if health.get('binance_configured') else 'DISCONNECTED'}
                </div>
            </div>
            <div class="status-item">
                <h3>Auto Trading</h3>
                <div class="status-value {'auto-trading-enabled' if auto_status.get('enabled') else 'auto-trading-disabled'}">
                    {'ENABLED' if auto_status.get('enabled') else 'DISABLED'}
                </div>
            </div>
        </div>
        
        <div class="content">
            <!-- Account Overview -->
            <div class="section">
                <h2>💰 Account Overview</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Total Balance</h4>
                        <div class="metric-value">
                            ${account_info.get('total_wallet_balance', 0):,.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h4>BTC Balance</h4>
                        <div class="metric-value">
                            {account_info.get('btc_balance', 0):.6f} BTC
                        </div>
                    </div>
                    <div class="metric-card">
                        <h4>USDT Balance</h4>
                        <div class="metric-value">
                            ${account_info.get('usdt_balance', 0):,.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h4>Current BTC Price</h4>
                        <div class="metric-value">
                            ${account_info.get('current_btc_price', 0):,.2f}
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Trading Performance -->
            <div class="section">
                <h2>📊 Trading Performance</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Total P&L</h4>
                        <div class="metric-value {'positive' if performance.get('total_pnl', 0) >= 0 else 'negative'}">
                            ${performance.get('total_pnl', 0):,.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h4>Win Rate</h4>
                        <div class="metric-value">
                            {performance.get('win_rate_percent', 0):.1f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <h4>Total Trades</h4>
                        <div class="metric-value">
                            {performance.get('total_trades', 0)}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h4>Total Return</h4>
                        <div class="metric-value {'positive' if performance.get('total_return_percent', 0) >= 0 else 'negative'}">
                            {performance.get('total_return_percent', 0):.2f}%
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- ML Signal -->
            <div class="section">
                <h2>🎯 Current ML Signal</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Signal</h4>
                        <div class="metric-value signal-{ml_signal.get('signal', 'hold').lower()}">
                            {ml_signal.get('signal', 'HOLD')}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h4>Confidence</h4>
                        <div class="metric-value">
                            {ml_signal.get('confidence', 0):.1%}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h4>Position Size</h4>
                        <div class="metric-value">
                            {ml_signal.get('position_size', 0):.1%}
                        </div>
                    </div>
                    <div class="metric-card">
                        <h4>Leverage</h4>
                        <div class="metric-value">
                            {ml_signal.get('leverage', 1):.1f}x
                        </div>
                    </div>
                </div>
                <p><strong>Reason:</strong> {ml_signal.get('reason', 'No reason provided')}</p>
            </div>
            
            <!-- Recent Trades -->
            <div class="section">
                <h2>📈 Recent Trades</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Side</th>
                            <th>Quantity</th>
                            <th>Price</th>
                            <th>Commission</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        # Add trade history rows
        for trade in trade_history[:10]:  # Show last 10 trades
            time_str = trade.get('time', '')
            try:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                formatted_time = dt.strftime("%m-%d %H:%M")
            except:
                formatted_time = time_str
            
            side_class = 'signal-buy' if trade.get('side') == 'BUY' else 'signal-sell'
            
            html += f"""
                        <tr>
                            <td>{formatted_time}</td>
                            <td class="{side_class}">{trade.get('side', '')}</td>
                            <td>{trade.get('quantity', 0):.6f}</td>
                            <td>${trade.get('price', 0):,.2f}</td>
                            <td>{trade.get('commission', 0):.6f}</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
            
            <!-- Current Positions -->
            <div class="section">
                <h2>💼 Current Positions</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Entry Price</th>
                            <th>Current Price</th>
                            <th>Amount</th>
                            <th>Unrealized P&L</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        # Add positions rows
        for position in positions:
            pnl_class = 'positive' if position.get('unrealized_pnl', 0) >= 0 else 'negative'
            
            html += f"""
                        <tr>
                            <td>{position.get('symbol', '')}</td>
                            <td>{position.get('side', '')}</td>
                            <td>${position.get('entry_price', 0):,.2f}</td>
                            <td>${position.get('current_price', 0):,.2f}</td>
                            <td>{position.get('amount', 0):.6f}</td>
                            <td class="{pnl_class}">${position.get('unrealized_pnl', 0):,.2f}</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
            
            <div class="refresh-info">
                <p>🔄 Report auto-refreshes every 30 seconds</p>
                <p>📊 Data stored in JSON files for historical analysis</p>
                <p>🤖 ML Algorithm: Optimized High-Return System (5x leverage, 63% confidence threshold)</p>
            </div>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
"""
        
        return html
    
    def save_html_report(self, html: str):
        """Save HTML report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_report_{timestamp}.html"
        filepath = self.reports_dir / filename
        
        # Also save as latest.html for easy access
        latest_filepath = self.reports_dir / "latest.html"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            
            with open(latest_filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            
            print(f"✅ HTML report saved to {filepath}")
            print(f"✅ Latest report saved to {latest_filepath}")
        except Exception as e:
            print(f"❌ Error saving HTML report: {e}")
    
    def run_report_generation(self):
        """Run the report generation process"""
        print("🔄 Fetching data from ML Trading API...")
        data = self.fetch_api_data()
        
        if "error" in data:
            print(f"❌ Error fetching data: {data['error']}")
            return
        
        print("💾 Saving data to JSON file...")
        self.save_data_to_json(data)
        
        print("📊 Generating HTML report...")
        html = self.generate_html_report(data)
        
        print("💾 Saving HTML report...")
        self.save_html_report(html)
        
        print("✅ Report generation completed!")
        print(f"🌐 Open the report: {self.reports_dir / 'latest.html'}")
    
    def run_continuous_monitoring(self, interval_minutes: int = 1):
        """Run continuous monitoring and report generation"""
        print(f"🔄 Starting continuous monitoring (every {interval_minutes} minute(s))")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                self.run_report_generation()
                print(f"⏰ Waiting {interval_minutes} minute(s) for next update...")
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")

def main():
    """Main function"""
    reporter = MLTradingReporter()
    
    print("🤖 ML Trading Report Generator")
    print("=" * 50)
    print("1. Generate single report")
    print("2. Start continuous monitoring (1 minute intervals)")
    print("3. Start continuous monitoring (5 minute intervals)")
    print("4. Start continuous monitoring (30 minute intervals)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        reporter.run_report_generation()
    elif choice == "2":
        reporter.run_continuous_monitoring(1)
    elif choice == "3":
        reporter.run_continuous_monitoring(5)
    elif choice == "4":
        reporter.run_continuous_monitoring(30)
    else:
        print("Invalid choice. Generating single report...")
        reporter.run_report_generation()

if __name__ == "__main__":
    main()
