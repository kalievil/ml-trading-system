import { useState, useEffect } from "react";

export interface Trade {
  id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  entry_price: number;
  exit_price: number | null;
  size: number;
  pnl: number | null;
  pnl_percent: number | null;
  confidence: number | null;
  entry_time: string;
  exit_time: string | null;
  status: 'open' | 'closed';
  stop_loss: number | null;
  take_profit: number | null;
  exit_reason: string | null;
  confirmation_details: {
    entryPrice: number;
    timestamp: string;
    leverage?: number;
    positionSizePct?: number;
    indicators: string[];
    technicalAnalysis: string;
    riskAssessment: string;
    marketCondition: string;
  } | null;
}

export interface Position {
  id: string;
  trade_id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  entry_price: number;
  current_price: number;
  size: number;
  pnl: number;
  pnl_percent: number;
  stop_loss: number | null;
  take_profit: number | null;
  confidence: number | null;
  opened_at: string;
}

export interface AlgorithmState {
  id: string;
  is_active: boolean;
  current_balance: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl: number;
  win_rate: number;
  last_update: string;
  confidence: number | null;
  status_message: string | null;
}

// Local storage helpers
const STORAGE_KEY = 'trading-app-data';

const saveToStorage = (data: { trades: Trade[], positions: Position[], algorithmState: AlgorithmState }) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch (error) {
    console.error('Failed to save to localStorage:', error);
  }
};

const loadFromStorage = (): { trades: Trade[], positions: Position[], algorithmState: AlgorithmState } | null => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (error) {
    console.error('Failed to load from localStorage:', error);
  }
  return null;
};

// Create initial empty state
const createInitialState = (): { trades: Trade[], positions: Position[], algorithmState: AlgorithmState } => {
  const now = new Date().toISOString();
  
  const algorithmState: AlgorithmState = {
    id: 'algo-1',
    is_active: true,
    current_balance: 10000.00,
    total_trades: 0,
    winning_trades: 0,
    losing_trades: 0,
    total_pnl: 0,
    win_rate: 0,
    last_update: now,
    confidence: 0,
    status_message: 'Algorithm ready - waiting for trading signals'
  };

  return {
    trades: [],
    positions: [],
    algorithmState
  };
};

export const useTradingData = () => {
  const [algorithmState, setAlgorithmState] = useState<AlgorithmState | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Always start with clean empty state (ignore any stored mock data)
    const loadData = () => {
      setIsLoading(true);
      
      // Clear any existing localStorage data to ensure clean start
      localStorage.removeItem(STORAGE_KEY);
      
      // Always create initial empty state
      const initialState = createInitialState();
      setAlgorithmState(initialState.algorithmState);
      setPositions(initialState.positions);
      setTrades(initialState.trades);
      saveToStorage(initialState);
      
      setIsLoading(false);
    };

    loadData();

    // Simulate real-time updates every 30 seconds (only if there are active positions)
    const interval = setInterval(() => {
      setPositions(currentPositions => {
        if (currentPositions.length === 0) return currentPositions;
        
        const updatedPositions = currentPositions.map(pos => {
          const priceChange = (Math.random() - 0.5) * 0.02; // ±1% price change
          const newPrice = pos.current_price * (1 + priceChange);
          const pnl = pos.side === 'LONG' 
            ? (newPrice - pos.entry_price) * pos.size
            : (pos.entry_price - newPrice) * pos.size;
          const pnlPercent = (pnl / (pos.entry_price * pos.size)) * 100;
          
          return {
            ...pos,
            current_price: newPrice,
            pnl,
            pnl_percent: pnlPercent
          };
        });
        
        // Update algorithm state with new P&L
        setAlgorithmState(prev => {
          if (!prev) return prev;
          
          const totalPnL = updatedPositions.reduce((sum, pos) => sum + pos.pnl, 0);
          const updated = {
            ...prev,
            current_balance: 10000 + totalPnL,
            total_pnl: totalPnL,
            last_update: new Date().toISOString(),
          };
          
          // Save updated data
          const currentData = { trades, positions: updatedPositions, algorithmState: updated };
          saveToStorage(currentData);
          
          return updated;
        });
        
        return updatedPositions;
      });
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  return {
    algorithmState,
    positions,
    trades: trades.filter(t => t.status === 'closed'),
    recentTrades: trades.slice(0, 5),
    isLoading,
  };
};