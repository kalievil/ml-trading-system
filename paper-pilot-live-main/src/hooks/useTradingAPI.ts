import { useState, useEffect } from "react";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface TradingSignal {
  symbol: string;
  side?: 'LONG' | 'SHORT';  // Optional for ML signals
  signal?: 'BUY' | 'SELL' | 'HOLD';  // ML signal format
  confidence: number;
  entry_price?: number;  // Optional for ML signals
  current_price?: number;  // ML signal format
  stop_loss: number;
  take_profit: number;
  position_size: number;
  leverage: number;
  timestamp: string;
  reason?: string;  // ML signal reason
  volatility?: number;  // ML signal volatility
  algorithm?: string;  // ML algorithm name
  models_loaded?: boolean;  // ML models status
  trading_halted?: boolean;  // ML trading status
  indicators?: {
    rsi_14: number;
    macd: number;
    volatility: number;
    volume_ratio: number;
  };
}

export interface MarketData {
  symbol: string;
  price: number;
  timestamp: string;
  volume: number;
  high: number;
  low: number;
  open: number;
  close: number;
}

export interface AccountInfo {
  btc_balance: number;
  usdt_balance: number;
  account_type: string;
  can_trade: boolean;
  can_withdraw: boolean;
}

export const useTradingAPI = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Configure Binance API
  const configureBinance = async (apiKey: string, secretKey: string) => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/configure-binance`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ api_key: apiKey, api_secret: secretKey }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to configure Binance API');
      }

      const result = await response.json();
      setIsConnected(true);
      return result;
    } catch (err: any) {
      setError(err.message);
      setIsConnected(false);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Get market data
  const getMarketData = async (): Promise<MarketData> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/market-data`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get market data');
      }

      return await response.json();
    } catch (err: any) {
      setError(err.message);
      throw err;
    }
  };

  // Get trading signal
  const getTradingSignal = async (): Promise<TradingSignal | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/trading-signal`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get trading signal');
      }

      const result = await response.json();
      
      // Check if it's a signal or a message
      if (result.symbol) {
        return result as TradingSignal;
      } else {
        return null; // No signal generated
      }
    } catch (err: any) {
      setError(err.message);
      throw err;
    }
  };

  // Get ML signal (ML-powered algorithm)
  const getMLSignal = async (): Promise<TradingSignal | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/ml-signal`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get ML signal');
      }

      const result = await response.json();
      return result as TradingSignal;
    } catch (err: any) {
      setError(err.message);
      throw err;
    }
  };

  // Get test signal (simple algorithm)
  const getTestSignal = async (): Promise<TradingSignal | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/test-signal`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get test signal');
      }

      const result = await response.json();
      return result as TradingSignal;
    } catch (err: any) {
      setError(err.message);
      return null;
    }
  };

  // Execute test trade
  const executeTestTrade = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/test-execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to execute test trade');
      }

      return await response.json();
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Execute trade
  const executeTrade = async (symbol: string, side: 'LONG' | 'SHORT', quantity: number, price: number) => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/execute-trade`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol,
          side,
          quantity,
          price
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to execute trade');
      }

      return await response.json();
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Manual trade (BUY by USDT or SELL by BTC or percentage)
  const manualTrade = async (params: { side: 'BUY' | 'SELL'; amount_usdt?: number; amount_btc?: number; percentage?: number; }) => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE_URL}/api/manual-trade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to execute manual trade');
      }
      return await response.json();
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Get account info
  const getAccountInfo = async (): Promise<AccountInfo> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/account-info`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get account info');
      }

      return await response.json();
    } catch (err: any) {
      setError(err.message);
      throw err;
    }
  };

  // Get positions
  const getPositions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/positions`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get positions');
      }

      const raw = await response.json();
      // Backend returns balances as pseudo-positions; map and filter to BTC only
      const mapped = (raw || [])
        .filter((p: any) => (p.symbol || '').toUpperCase().startsWith('BTC'))
        .map((p: any) => {
          const symbol = p.symbol || 'BTCUSDT';
          const amount = Number(p.amount ?? 0);
          const entry = Number(p.entry_price ?? p.current_price ?? 0);
          const current = Number(p.current_price ?? entry);
          const upnl = Number(p.unrealized_pnl ?? (current - entry) * amount);
          const upnlPct = Number(p.unrealized_pnl_percent ?? (entry > 0 ? ((current - entry) / entry) * 100 : 0));
          return {
            symbol,
            side: (p.side as any) || 'LONG',
            amount,
            entry_price: entry,
            current_price: current,
            unrealized_pnl: upnl,
            unrealized_pnl_percent: upnlPct,
          };
        });
      return mapped;
    } catch (err: any) {
      setError(err.message);
      return [];
    }
  };

  // Get trade history
  const getTradeHistory = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/trade-history`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get trade history');
      }

      const raw = await response.json();
      // Map backend fields to UI-friendly shape expected by TradeHistory component
      const mapped = (raw || []).map((t: any, idx: number) => {
        const timeMs = typeof t.time === 'string' ? Date.parse(t.time) : (typeof t.time === 'number' ? t.time : Date.now());
        const qty = Number(t.quantity ?? t.qty ?? 0);
        const price = Number(t.price ?? 0);
        return {
          id: `${t.symbol || 'BTCUSDT'}-${timeMs}-${idx}`,
          symbol: t.symbol || 'BTCUSDT',
          side: t.side || (t.isBuyer ? 'BUY' : 'SELL'),
          quantity: qty,
          price: price,
          commission: Number(t.commission ?? 0),
          commission_asset: t.commission_asset || t.commissionAsset || 'USDT',
          time: timeMs,
          // Optional fields used by the table; backend doesn't provide these
          entry_price: undefined,
          exit_price: undefined,
          pnl: null,
          pnl_percent: null,
          total_value: qty * price,
        };
      });
      return mapped;
    } catch (err: any) {
      setError(err.message);
      return [];
    }
  };

  // Get trading performance metrics
  const getTradingPerformance = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/trading-performance`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get trading performance');
      }

      const raw = await response.json();
      // Normalize backend fields to the UI's expected shape
      const normalized = {
        ...raw,
        current_portfolio_value: raw.total_wallet_balance ?? raw.current_portfolio_value ?? 0,
        btc_balance: raw.current_btc_balance ?? raw.btc_balance ?? 0,
        usdt_balance: raw.current_usdt_balance ?? raw.usdt_balance ?? 0,
        current_price: raw.current_btc_price ?? raw.current_price ?? 0,
        total_return_percent: raw.total_return ?? raw.total_return_percent ?? 0,
      };
      return normalized;
    } catch (err: any) {
      setError(err.message);
      return null;
    }
  };

  // Close position
  const closePosition = async (symbol: string = 'BTCUSDT') => {
    try {
      setIsLoading(true);
      setError(null);

      // Use convert endpoint to sell BTC to USDT
      const response = await fetch(`${API_BASE_URL}/api/convert-btc-to-usdt`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ percent: 1.0 }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to close position');
      }

      const res = await response.json();
      // Normalize for UI toast in PositionsPanel
      return {
        btc_sold: res.sold_btc ?? 0,
        estimated_usdt_received: res.approx_usdt ?? 0,
        order_id: res.order_id,
        status: res.status,
      };
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Validate data consistency
  const validateData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/data-validation`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Data validation error:', error);
      throw error;
    }
  };

  // Health check
  const healthCheck = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      
      if (!response.ok) {
        throw new Error('API server not responding');
      }

      const health = await response.json();
      return health;
    } catch (err: any) {
      setError(err.message);
      return null;
    }
  };

  const getAutoTradingStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auto-trading-status`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Auto trading status error:', error);
      throw error;
    }
  };

  const enableAutoTrading = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/enable-auto-trading`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to enable auto trading');
      }

      return await response.json();
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const disableAutoTrading = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/disable-auto-trading`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to disable auto trading');
      }

      return await response.json();
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Get tracked positions (with SL/TP) from backend
  const getTrackedPositions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/open-positions`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get tracked positions');
      }
      const data = await response.json();
      return data?.tracked_positions ?? [];
    } catch (err: any) {
      setError(err.message);
      return [];
    }
  };

  // Check API connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const health = await healthCheck();
        if (health) {
          // Check if API is healthy and has credentials (either configured or from file)
          const hasCredentials = health.binance_configured || health.credentials_from_file;
          const shouldBeConnected = health.ml_system_ready && hasCredentials;
          
          // Only update state if connection status actually changed
          setIsConnected(prevConnected => {
            if (prevConnected !== shouldBeConnected) {
              return shouldBeConnected;
            }
            return prevConnected;
          });
        } else {
          setIsConnected(false);
        }
      } catch (error) {
        // Only set to false if we were previously connected
        setIsConnected(prevConnected => prevConnected ? false : prevConnected);
      }
    };

    checkConnection();
    
    // Check connection every 5 seconds for responsive updates
    const interval = setInterval(checkConnection, 5000);
    return () => clearInterval(interval);
  }, []);

  return {
    isConnected,
    isLoading,
    error,
    configureBinance,
    getMarketData,
    getTradingSignal,
    getMLSignal,
    getTestSignal,
    executeTrade,
    executeTestTrade,
    manualTrade,
    getAccountInfo,
    getPositions,
    getTradeHistory,
    getTradingPerformance,
    closePosition,
    validateData,
    getAutoTradingStatus,
    enableAutoTrading,
    disableAutoTrading,
    getTrackedPositions,
    healthCheck,
  };
};
