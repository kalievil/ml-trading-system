import { useState, useEffect } from "react";

const API_BASE_URL = "http://localhost:8000";

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

      return await response.json();
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

      return await response.json();
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

      return await response.json();
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

      const response = await fetch(`${API_BASE_URL}/api/close-position`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symbol }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to close position');
      }

      return await response.json();
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
    getAccountInfo,
    getPositions,
    getTradeHistory,
    getTradingPerformance,
    closePosition,
    validateData,
    getAutoTradingStatus,
    enableAutoTrading,
    disableAutoTrading,
    healthCheck,
  };
};
