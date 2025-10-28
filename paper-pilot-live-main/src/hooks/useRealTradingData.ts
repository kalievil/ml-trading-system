import { useState, useEffect } from "react";
import { useTradingAPI } from "./useTradingAPI";

export interface RealTrade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  commission: number;
  commission_asset: string;
  time: number;
  is_buyer: boolean;
  is_maker: boolean;
}

export interface RealPosition {
  symbol: string;
  side: 'LONG' | 'SHORT';
  amount: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
}

export interface RealAccountInfo {
  btc_balance: number;
  usdt_balance: number;
  account_type: string;
  can_trade: boolean;
  can_withdraw: boolean;
  total_wallet_balance: number;
}

export interface TradingPerformance {
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
  current_portfolio_value: number;
  btc_balance: number;
  usdt_balance: number;
  current_price: number;
  win_rate_percent: number;
  total_trades: number;
  completed_trades: number;
  profitable_trades: number;
  losing_trades: number;
  average_win: number;
  average_loss: number;
  total_return_percent: number;
  daily_return_percent: number;
  portfolio_change_24h: number;
  btc_inventory: number;
  avg_buy_price: number;
  trade_history: any[];
}

export const useRealTradingData = () => {
  const [accountInfo, setAccountInfo] = useState<RealAccountInfo | null>(null);
  const [positions, setPositions] = useState<RealPosition[]>([]);
  const [tradeHistory, setTradeHistory] = useState<RealTrade[]>([]);
  const [tradingPerformance, setTradingPerformance] = useState<TradingPerformance | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const { isConnected, getAccountInfo, getPositions, getTradeHistory, getTradingPerformance, closePosition } = useTradingAPI();

  const fetchRealData = async () => {
    if (!isConnected) {
      return; // Don't change loading state during background updates
    }

    try {
      // Silent background update - no loading states, no error states
      // Fetch all data in parallel
      const [account, positions, trades, performance] = await Promise.all([
        getAccountInfo(),
        getPositions(),
        getTradeHistory(),
        getTradingPerformance()
      ]);

      // Silent updates - no loading states, no error states
      setAccountInfo(account);
      setPositions(positions);
      setTradeHistory(trades);
      setTradingPerformance(performance);
    } catch (err: any) {
      // Silent error handling - don't disrupt UI or change states
      console.error('Background data fetch error:', err);
    }
  };

  const fetchRealDataWithLoading = async () => {
    if (!isConnected) {
      setIsLoading(false);
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      // Fetch all data in parallel
      const [account, positions, trades, performance] = await Promise.all([
        getAccountInfo(),
        getPositions(),
        getTradeHistory(),
        getTradingPerformance()
      ]);

      setAccountInfo(account);
      setPositions(positions);
      setTradeHistory(trades);
      setTradingPerformance(performance);
    } catch (err: any) {
      setError(err.message);
      console.error('Error fetching real trading data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // Initial load with loading state
    fetchRealDataWithLoading();
    
    // Background updates every 3 seconds - completely silent
    const interval = setInterval(fetchRealData, 3000);
    return () => clearInterval(interval);
  }, [isConnected]);

  // Refresh function for manual updates
  const refreshData = () => {
    fetchRealDataWithLoading();
  };

  // Close position function
  const closePositionAndRefresh = async (symbol: string = 'BTCUSDT') => {
    try {
      const result = await closePosition(symbol);
      // Refresh data after closing position with loading state
      await fetchRealDataWithLoading();
      return result;
    } catch (error) {
      throw error;
    }
  };

  return {
    accountInfo,
    positions,
    tradeHistory,
    tradingPerformance,
    isLoading,
    error,
    refreshData,
    closePositionAndRefresh,
    isConnected
  };
};
