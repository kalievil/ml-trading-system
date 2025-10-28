#!/usr/bin/env python3
"""
Optimized High-Return Risk-Managed Freqtrade Strategy
Adapted from complete_high_return_optimized.py for Freqtrade backtesting

Key Features:
- Dynamic leverage based on signal confidence (1x-5x)
- Custom stop-loss at 1.8%
- Take-profit at 1.5%
- Support for long and short positions
- Volatility filtering
- Risk management similar to original system
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    informative,
    DecimalParameter,
    IntParameter,
    RealParameter,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib


class OptimizedHighReturnStrategy(IStrategy):
    """
    Optimized High-Return Risk-Managed Trading Strategy for Freqtrade
    
    Original Logic Adapted From:
    - Uses similar indicators (RSI, MACD, ATR, volatility)
    - Implements dynamic leverage (1x-5x) based on signal strength
    - Custom stop-loss and take-profit logic
    - Long and short support
    """
    
    # Strategy interface version
    INTERFACE_VERSION = 3
    
    # Enable short positions
    can_short = True
    
    # Timeframe
    timeframe = '5m'
    
    # Startup candles needed
    startup_candle_count: int = 200
    
    # Trailing stoploss - disabled, using custom logic
    trailing_stop = False
    
    # Minimal ROI configuration
    minimal_roi = {
        "0": 0.15,  # Target 1.5% profit for quick exits
        "15": 0.10,  # Reduce to 1% after 15 candles
        "30": 0.05,  # Reduce to 0.5% after 30 candles
    }
    
    # Base stoploss - will be overridden by custom logic
    stoploss = -0.018  # 1.8% stoploss
    
    # Use custom stoploss
    use_custom_stoploss = True
    
    # Exit configuration
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Process only new candles
    process_only_new_candles = True
    
    # Hyperoptable parameters
    buy_rsi = IntParameter(low=30, high=60, default=40, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=40, high=70, default=60, space='sell', optimize=True, load=True)
    
    # Volatility and signal strength parameters
    min_confidence = DecimalParameter(0.60, 0.75, default=0.63, space='buy', optimize=True, load=True)
    max_volatility = DecimalParameter(0.015, 0.025, default=0.022, space='buy', optimize=True, load=True)
    
    # Leverage parameters
    max_leverage = DecimalParameter(3.0, 5.0, default=5.0, space='buy', optimize=False, load=True)
    min_leverage_confidence = DecimalParameter(0.70, 0.80, default=0.73, space='buy', optimize=False, load=True)
    
    # Position sizing
    base_position_size = DecimalParameter(0.25, 0.35, default=0.30, space='buy', optimize=False, load=True)
    max_position_size = DecimalParameter(0.30, 0.40, default=0.35, space='buy', optimize=False, load=True)
    
    # Order types
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    
    # Time in force
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all required indicators for the strategy
        """
        
        # Momentum Indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_21'] = ta.RSI(dataframe, timeperiod=21)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['normalized_atr'] = dataframe['atr'] / dataframe['close']
        
        # Volatility calculation
        dataframe['returns'] = dataframe['close'].pct_change()
        dataframe['volatility'] = dataframe['returns'].rolling(window=20).std() * np.sqrt(288)  # Annualized
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lowerband']) / \
                                   (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        
        # Moving Averages
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        # TEMA for trend
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        
        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe)
        
        # Volume indicators
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume'].rolling(window=20).mean()
        
        # Support/Resistance levels (simplified using recent price action)
        dataframe['recent_high'] = dataframe['high'].rolling(window=20).max()
        dataframe['recent_low'] = dataframe['low'].rolling(window=20).min()
        dataframe['distance_to_support'] = (dataframe['close'] - dataframe['recent_low']) / dataframe['close']
        dataframe['distance_to_resistance'] = (dataframe['recent_high'] - dataframe['close']) / dataframe['close']
        
        # Price momentum
        dataframe['price_momentum_1'] = dataframe['close'].pct_change(periods=1)
        dataframe['price_momentum_3'] = dataframe['close'].pct_change(periods=3)
        dataframe['price_momentum_5'] = dataframe['close'].pct_change(periods=5)
        
        # Breakout detection
        dataframe['higher_high'] = (dataframe['close'] > dataframe['high'].shift(1)).astype(int)
        dataframe['lower_low'] = (dataframe['close'] < dataframe['low'].shift(1)).astype(int)
        
        # Calculate signal confidence - this simulates the ML model's confidence score
        dataframe['signal_confidence'] = self._calculate_signal_confidence(dataframe)
        
        # Volatility ratio for filtering
        dataframe['volatility_ratio'] = dataframe['volatility'] / 0.02
        
        # Store original price for later use
        dataframe['original_price'] = dataframe['close']
        
        return dataframe
    
    def _calculate_signal_confidence(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate signal confidence based on multiple indicators
        Simulates the ML model's confidence scoring
        """
        
        # Start with neutral confidence
        confidence = pd.Series(0.5, index=dataframe.index)
        
        # RSI-based confidence
        rsi_signal = np.where(
            (dataframe['rsi'] < 40) & (dataframe['rsi'] > 30),
            0.7,  # Good long signal
            np.where(
                (dataframe['rsi'] > 60) & (dataframe['rsi'] < 70),
                0.7,  # Good short signal
                0.5  # Neutral
            )
        )
        
        # MACD-based confidence
        macd_signal = np.where(
            (dataframe['macd'] > dataframe['macdsignal']) & (dataframe['macd'] > 0),
            0.75,
            np.where(
                (dataframe['macd'] < dataframe['macdsignal']) & (dataframe['macd'] < 0),
                0.75,
                0.5
            )
        )
        
        # Trend strength from ADX
        adx_signal = np.where(
            dataframe['adx'] > 25,
            dataframe['adx'] / 50,  # Scale ADX to 0-1
            0.4
        )
        
        # Volume confirmation
        volume_signal = np.where(
            dataframe['volume_ratio'] > 1.2,
            1.0,  # High volume confirmation
            np.where(
                dataframe['volume_ratio'] < 0.8,
                0.3,  # Low volume
                0.5
            )
        )
        
        # Trend confirmation
        trend_signal = np.where(
            dataframe['close'] > dataframe['ema_20'],
            0.75,  # Uptrend
            np.where(
                dataframe['close'] < dataframe['ema_20'],
                0.75,  # Downtrend
                0.5
            )
        )
        
        # Combine all signals with weights
        confidence = (
            0.25 * rsi_signal +
            0.25 * macd_signal +
            0.15 * adx_signal +
            0.15 * volume_signal +
            0.20 * trend_signal
        )
        
        # Ensure confidence is between 0 and 1
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return confidence
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry logic for long and short positions
        """
        
        # Long entry conditions
        dataframe.loc[
            (
                # Core conditions
                (dataframe['signal_confidence'] >= self.min_confidence.value) &  # Minimum confidence
                (dataframe['volatility'] <= self.max_volatility.value) &  # Volatility filter
                (qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value)) &  # RSI buy signal
                (dataframe['macd'] > dataframe['macdsignal']) &  # MACD bullish
                (dataframe['close'] < dataframe['bb_upperband']) &  # Not overbought
                (dataframe['volume_ratio'] > 1.0) &  # Volume confirmation
                (dataframe['adx'] > 20) &  # Some trend strength
                (dataframe['volume'] > 0)
            ),
            'enter_long',
        ] = 1
        
        # Short entry conditions
        dataframe.loc[
            (
                # Core conditions
                (dataframe['signal_confidence'] >= self.min_confidence.value) &  # Minimum confidence
                (dataframe['volatility'] <= self.max_volatility.value) &  # Volatility filter
                (qtpylib.crossed_above(dataframe['rsi'], 100 - self.buy_rsi.value)) &  # RSI sell signal
                (dataframe['macd'] < dataframe['macdsignal']) &  # MACD bearish
                (dataframe['close'] > dataframe['bb_lowerband']) &  # Not oversold
                (dataframe['volume_ratio'] > 1.0) &  # Volume confirmation
                (dataframe['adx'] > 20) &  # Some trend strength
                (dataframe['volume'] > 0)
            ),
            'enter_short',
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit logic for long and short positions
        """
        
        # Exit long positions
        dataframe.loc[
            (
                (
                    (dataframe['rsi'] > self.sell_rsi.value)  # Overbought
                    | (dataframe['macd'] < dataframe['macdsignal'])  # MACD turns bearish
                    | (
                        (dataframe['close'] > dataframe['bb_upperband'])
                        & (qtpylib.crossed_below(dataframe['close'], dataframe['bb_upperband']))
                    )  # Exit BB
                )
                & (dataframe['volume'] > 0)
            ),
            'exit_long',
        ] = 1
        
        # Exit short positions
        dataframe.loc[
            (
                (
                    (dataframe['rsi'] < (100 - self.sell_rsi.value))  # Oversold
                    | (dataframe['macd'] > dataframe['macdsignal'])  # MACD turns bullish
                    | (
                        (dataframe['close'] < dataframe['bb_lowerband'])
                        & (qtpylib.crossed_above(dataframe['close'], dataframe['bb_lowerband']))
                    )  # Exit BB
                )
                & (dataframe['volume'] > 0)
            ),
            'exit_short',
        ] = 1
        
        return dataframe
    
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs
    ) -> float:
        """
        Custom leverage calculation based on signal confidence
        """
        
        # Get the current dataframe to access signal confidence
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) > 0:
            # Get the latest signal confidence
            latest_confidence = dataframe['signal_confidence'].iloc[-1]
            
            # Calculate dynamic leverage based on confidence
            if latest_confidence >= self.min_leverage_confidence.value:
                # Use the configured max leverage
                return float(min(self.max_leverage.value, max_leverage))
            elif latest_confidence >= 0.65:
                # Medium confidence - use 3x leverage
                return float(min(3.0, max_leverage))
            else:
                # Low confidence - use 2x leverage
                return float(min(2.0, max_leverage))
        
        # Default to 2x if we can't determine confidence
        return 2.0
    
    def custom_stoploss(
        self,
        pair: str,
        trade: 'Trade',
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs
    ) -> float:
        """
        Custom stoploss logic - maintains 1.8% stop-loss
        """
        
        # Use the original stoploss of 1.8%
        # This creates a hard stop at entry - 1.8%
        return self.stoploss


