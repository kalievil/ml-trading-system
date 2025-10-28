#!/usr/bin/env python3

"""
Custom Backtesting Wrapper for OptimizedHighReturnSystem
Integrates the custom algorithm with Freqtrade's backtesting infrastructure
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, Tuple

# Import the original optimized system
import sys
sys.path.append('.')
from complete_high_return_optimized import OptimizedHighReturnSystem

logger = logging.getLogger(__name__)


class FreqtradeBacktestWrapper:
    """
    Wrapper to run OptimizedHighReturnSystem and generate Freqtrade-compatible results
    """
    
    def __init__(self, config_path: str = "user_data/config_optimized_testnet.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.results_dir = Path("user_data/backtest_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_config(self) -> Dict[str, Any]:
        """Load Freqtrade configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def convert_to_freqtrade_format(self, performance: Dict[str, Any], portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OptimizedHighReturnSystem results to Freqtrade format"""
        
        # Create Freqtrade-compatible results structure
        freqtrade_results = {
            "strategy": {
                "OptimizedXGBoostStrategy": {
                    "trades": len(portfolio.get('trades', [])),
                    "profit_total": performance.get('total_return', 0),
                    "profit_total_pct": performance.get('total_return', 0),
                    "profit_mean": 0,
                    "profit_mean_pct": 0,
                    "profit_sum": performance.get('final_value', 10000) - 10000,
                    "profit_sum_pct": performance.get('total_return', 0),
                    "profit_total_abs": performance.get('final_value', 10000) - 10000,
                    "profit_total_abs_pct": performance.get('total_return', 0),
                    "winning_trades": performance.get('winning_trades', 0),
                    "losing_trades": performance.get('completed_trades', 0) - performance.get('winning_trades', 0),
                    "win_rate": performance.get('win_rate', 0),
                    "drawdown": 0,
                    "drawdown_abs": 0,
                    "drawdown_start": None,
                    "drawdown_end": None,
                    "trades": self.convert_trades_to_freqtrade_format(portfolio.get('trades', [])),
                    "pairlist": ["BTC/USDT"],
                    "best_pair": "BTC/USDT",
                    "worst_pair": "BTC/USDT"
                }
            },
            "strategy_comparison": [
                {
                    "key": "OptimizedXGBoostStrategy",
                    "trades": len(portfolio.get('trades', [])),
                    "profit_total_pct": performance.get('total_return', 0),
                    "profit_total_abs": performance.get('final_value', 10000) - 10000,
                    "win_rate": performance.get('win_rate', 0),
                    "avg_profit_pct": 0,
                    "avg_profit_abs": 0,
                    "total_profit_pct": performance.get('total_return', 0),
                    "total_profit_abs": performance.get('final_value', 10000) - 10000,
                    "avg_duration": "0:00:00",
                    "max_drawdown": 0,
                    "max_drawdown_abs": 0,
                    "profit_factor": 0,
                    "sharpe": 0,
                    "calmar": 0,
                    "sortino": 0,
                    "expectancy": 0,
                    "expectancy_ratio": 0,
                    "sqn": 0,
                    "cagr": 0,
                    "cagr_pct": 0
                }
            ],
            "metadata": {
                "OptimizedXGBoostStrategy": {
                    "run_id": f"optimized_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "backtest_start_time": datetime.now().isoformat(),
                    "timeframe": "5m",
                    "backtest_start_ts": int((datetime.now() - timedelta(days=30)).timestamp()),
                    "backtest_end_ts": int(datetime.now().timestamp()),
                    "strategy": "OptimizedXGBoostStrategy",
                    "strategy_version": "1.0.0",
                    "notes": "Custom XGBoost Optimized Strategy Backtest"
                }
            }
        }
        
        return freqtrade_results
    
    def convert_trades_to_freqtrade_format(self, trades: list) -> list:
        """Convert trades to Freqtrade format"""
        freqtrade_trades = []
        
        # Group trades by entry/exit pairs
        trade_pairs = []
        current_trade = None
        
        for trade in trades:
            if trade['action'] in ['BUY', 'SELL']:
                current_trade = {
                    'entry': trade,
                    'exit': None
                }
            elif trade['action'] == 'CLOSE' and current_trade:
                current_trade['exit'] = trade
                trade_pairs.append(current_trade)
                current_trade = None
        
        # Convert to Freqtrade format
        for i, trade_pair in enumerate(trade_pairs):
            entry = trade_pair['entry']
            exit_trade = trade_pair['exit']
            
            if entry and exit_trade:
                # Calculate PnL
                if entry['action'] == 'BUY':
                    pnl = (exit_trade['price'] - entry['price']) * entry['shares']
                else:  # SELL
                    pnl = (entry['price'] - exit_trade['price']) * entry['shares']
                
                freqtrade_trade = {
                    "trade_id": i + 1,
                    "pair": "BTC/USDT",
                    "is_short": entry['action'] == 'SELL',
                    "amount": entry['shares'],
                    "stake_amount": entry['shares'] * entry['price'],
                    "max_stake_amount": entry['shares'] * entry['price'],
                    "open_date": datetime.fromtimestamp(entry['timestamp']).isoformat(),
                    "close_date": datetime.fromtimestamp(exit_trade['timestamp']).isoformat(),
                    "open_rate": entry['price'],
                    "close_rate": exit_trade['price'],
                    "close_profit": pnl,
                    "close_profit_pct": (pnl / (entry['shares'] * entry['price'])) * 100,
                    "close_profit_abs": pnl,
                    "stake_currency": "USDT",
                    "base_currency": "BTC",
                    "quote_currency": "USDT",
                    "fee_open": entry.get('commission', 0),
                    "fee_close": exit_trade.get('commission', 0),
                    "fee_open_cost": entry.get('commission', 0),
                    "fee_close_cost": exit_trade.get('commission', 0),
                    "fee_open_currency": "USDT",
                    "fee_close_currency": "USDT",
                    "open_order_id": f"entry_{i}",
                    "close_order_id": f"exit_{i}",
                    "exit_reason": exit_trade.get('reason', 'unknown'),
                    "initial_stop_loss_abs": entry['price'] * 0.982,  # 1.8% stop loss
                    "initial_stop_loss_pct": -1.8,
                    "stop_loss_abs": entry['price'] * 0.982,
                    "stop_loss_pct": -1.8,
                    "initial_stop_loss_ratio": -0.018,
                    "stop_loss_ratio": -0.018,
                    "min_rate": min(entry['price'], exit_trade['price']),
                    "max_rate": max(entry['price'], exit_trade['price']),
                    "leverage": entry.get('leverage', 1.0),
                    "interest_rate": 0,
                    "liquidation_price": None,
                    "is_open": False,
                    "enter_tag": f"confidence_{entry.get('confidence', 0.5):.3f}",
                    "exit_tag": None,
                    "funding_fees": 0,
                    "trading_mode": "futures",
                    "margin_mode": "isolated"
                }
                
                freqtrade_trades.append(freqtrade_trade)
        
        return freqtrade_trades
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save backtest results in Freqtrade format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Backtest results saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return ""
    
    def run_backtest(self, use_sample: bool = False) -> Tuple[Dict[str, Any], str]:
        """Run the OptimizedHighReturnSystem backtest and convert results"""
        logger.info("🚀 Starting OptimizedHighReturnSystem backtest...")
        
        try:
            # Initialize the optimized system
            system = OptimizedHighReturnSystem(max_leverage=5)
            
            # Try to load data from existing Freqtrade format first
            futures_data_file = Path("user_data/data/binance/futures/BTC_USDT_USDT-5m-futures.feather")
            if futures_data_file.exists():
                logger.info("📊 Loading data from existing Freqtrade format...")
                df = pd.read_feather(futures_data_file)
                # Convert Freqtrade format to the format expected by OptimizedHighReturnSystem
                df = df.reset_index()
                df['timestamp'] = df['date']
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"📊 Loaded {len(df)} records from Freqtrade data")
                
                # Override the load_binance_data method temporarily
                system.load_binance_data = lambda timeframe='5m': df
            else:
                # Fallback to original method
                df = system.load_binance_data()
                if df is None:
                    logger.error("Failed to load data for backtest")
                    return None, None
            
            # Run backtest
            performance, portfolio = system.run_backtest(use_sample=use_sample)
            
            if performance is None or portfolio is None:
                logger.error("❌ Backtest failed")
                return {}, ""
            
            # Convert to Freqtrade format
            freqtrade_results = self.convert_to_freqtrade_format(performance, portfolio)
            
            # Save results
            results_file = self.save_results(freqtrade_results)
            
            logger.info("✅ Backtest completed successfully")
            logger.info(f"📊 Total Return: {performance.get('total_return', 0):.2f}%")
            logger.info(f"📊 Win Rate: {performance.get('win_rate', 0):.2f}%")
            logger.info(f"📊 Total Trades: {performance.get('total_trades', 0)}")
            
            return freqtrade_results, results_file
            
        except Exception as e:
            logger.error(f"❌ Backtest failed with error: {e}")
            return {}, ""
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a summary report of the backtest results"""
        if not results or 'strategy' not in results:
            return "No results to report"
        
        strategy_results = results['strategy'].get('OptimizedXGBoostStrategy', {})
        
        report = f"""
🏆 OPTIMIZED XGBOOST STRATEGY BACKTEST RESULTS
{'='*60}
📊 Performance Metrics:
   • Total Return: {strategy_results.get('profit_total_pct', 0):.2f}%
   • Win Rate: {strategy_results.get('win_rate', 0):.2f}%
   • Total Trades: {strategy_results.get('trades', 0)}
   • Winning Trades: {strategy_results.get('winning_trades', 0)}
   • Losing Trades: {strategy_results.get('losing_trades', 0)}
   • Final Value: ${10000 + strategy_results.get('profit_total_abs', 0):.2f}

📈 Strategy Details:
   • Strategy: OptimizedXGBoostStrategy
   • Timeframe: 5m
   • Max Leverage: 5x
   • Stop Loss: 1.8%
   • Take Profit: 1.5%

🔧 Model Configuration:
   • XGBoost Ensemble: 3 models
   • Min Confidence: 63%
   • Position Size: 30-35%
   • Volatility Filter: 2.2%

{'='*60}
        """
        
        return report


def main():
    """Main function to run the custom backtest"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("🚀 Starting Custom Freqtrade Backtest Wrapper")
    
    # Initialize wrapper
    wrapper = FreqtradeBacktestWrapper()
    
    # Run backtest
    results, results_file = wrapper.run_backtest(use_sample=True)  # Use sample for testing
    
    if results:
        # Generate and print summary
        summary = wrapper.generate_summary_report(results)
        print(summary)
        
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info("🎉 Custom backtest completed successfully!")
    else:
        logger.error("❌ Custom backtest failed!")


if __name__ == "__main__":
    main()
