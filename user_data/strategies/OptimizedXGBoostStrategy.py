#!/usr/bin/env python3

"""
Optimized XGBoost Strategy for Freqtrade
Integrates the OptimizedHighReturnSystem algorithm into Freqtrade's architecture
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import joblib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from freqtrade.strategy.parameters import CategoricalParameter
from freqtrade.persistence import Trade
import talib.abstract as ta
from technical import qtpylib

logger = logging.getLogger(__name__)


class OptimizedXGBoostStrategy(IStrategy):
    """
    Optimized XGBoost Strategy integrating the OptimizedHighReturnSystem algorithm
    """
    
    # Strategy interface version
    INTERFACE_VERSION = 3
    
    # Optimal timeframe for the strategy
    timeframe = '5m'
    
    # Can this strategy go short?
    can_short: bool = True
    
    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.015  # 1.5% take profit (optimized)
    }
    
    # Optimal stoploss designed for the strategy
    stoploss = -0.018  # 1.8% stop loss (optimized)
    
    # Trailing stoploss
    trailing_stop = False
    use_custom_stoploss = True
    
    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True
    
    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200
    
    # Strategy parameters
    min_confidence_for_trade = DecimalParameter(0.60, 0.70, default=0.63, space="buy")
    min_confidence_for_leverage = DecimalParameter(0.70, 0.80, default=0.73, space="buy")
    min_confidence_for_max_leverage = DecimalParameter(0.85, 0.95, default=0.88, space="buy")
    max_leverage = IntParameter(3, 7, default=5, space="buy")
    base_position_size = DecimalParameter(0.25, 0.35, default=0.30, space="buy")
    max_position_size = DecimalParameter(0.30, 0.40, default=0.35, space="buy")
    max_volatility_threshold = DecimalParameter(0.020, 0.025, default=0.022, space="buy")
    
    # Model and scaler storage
    models: Dict[str, Any] = {}
    scaler: Optional[RobustScaler] = None
    feature_columns: list = []
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.load_models()
        
    def load_models(self):
        """Load pre-trained XGBoost models and scaler"""
        try:
            models_path = Path("user_data/models")
            if not models_path.exists():
                logger.warning("Models directory not found. Will train models on first run.")
                return
                
            # Load scaler
            scaler_file = models_path / "scaler.pkl"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
                logger.info("✅ Loaded scaler")
            
            # Load models
            self.models = {}
            model_files = {
                'xgboost_performance': 'xgboost_performance.pkl',
                'xgboost_aggressive': 'xgboost_aggressive.pkl', 
                'xgboost_balanced': 'xgboost_balanced.pkl'
            }
            
            for name, filename in model_files.items():
                model_file = models_path / filename
                if model_file.exists():
                    self.models[name] = joblib.load(model_file)
                    logger.info(f"✅ Loaded {name} model")
            
            # Load ensemble weights (if available) or use equal weights
            weights_file = models_path / "ensemble_weights.json"
            if weights_file.exists():
                try:
                    with open(weights_file, 'r') as f:
                        self.ensemble_weights = json.load(f)
                    logger.info("✅ Loaded ensemble weights")
                except Exception as e:
                    logger.warning(f"⚠️ Error loading ensemble weights: {e}")
                    self.ensemble_weights = {name: 1.0/len(self.models) for name in self.models.keys()}
            else:
                # Use equal weights as fallback
                self.ensemble_weights = {name: 1.0/len(self.models) for name in self.models.keys()}
                logger.info("📊 Using equal ensemble weights")
            
            # Load feature columns
            features_file = models_path / "feature_columns.json"
            if features_file.exists():
                with open(features_file, 'r') as f:
                    self.feature_columns = json.load(f)
                logger.info(f"✅ Loaded {len(self.feature_columns)} feature columns")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def save_models(self):
        """Save trained models and scaler"""
        try:
            models_path = Path("user_data/models")
            models_path.mkdir(exist_ok=True)
            
            # Save scaler
            if self.scaler:
                joblib.dump(self.scaler, models_path / "scaler.pkl")
            
            # Save models
            for name, model in self.models.items():
                joblib.dump(model, models_path / f"{name}.pkl")
            
            # Save feature columns
            with open(models_path / "feature_columns.json", 'w') as f:
                json.dump(self.feature_columns, f)
            
            # Save ensemble weights
            if hasattr(self, 'ensemble_weights'):
                with open(models_path / "ensemble_weights.json", 'w') as f:
                    json.dump(self.ensemble_weights, f)
                logger.info("✅ Ensemble weights saved")
                
            logger.info("✅ Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def calculate_time_weights(self, lookback_period):
        """Calculate exponential time decay weights"""
        weights = np.exp(np.linspace(-2, 0, lookback_period))
        return weights / weights.sum()

    def calculate_fast_market_profile(self, df, current_idx):
        """Calculate fast market profile using quantiles and pivot points"""
        lookback_period = 200
        atr_period = 14
        kde_bandwidth_multiplier = 0.3
        min_prominence = 0.05
        min_distance_between_levels = 0.003
        
        if current_idx < lookback_period:
            return None, None
        
        recent_data = df.iloc[current_idx - lookback_period:current_idx]
        prices = recent_data['close'].values
        volumes = recent_data['volume'].values
        
        time_weights = self.calculate_time_weights(len(prices))
        
        high_low = recent_data['high'] - recent_data['low']
        high_close = np.abs(recent_data['high'] - recent_data['close'].shift(1))
        low_close = np.abs(recent_data['low'] - recent_data['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=atr_period).mean().iloc[-1]
        
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        price_levels = np.quantile(prices, quantiles)
        
        densities = []
        for level in price_levels:
            distance = np.abs(prices - level)
            bandwidth = atr * kde_bandwidth_multiplier
            
            weights = np.exp(-0.5 * (distance / bandwidth) ** 2) * volumes * time_weights
            density = np.sum(weights)
            densities.append(density)
        
        return price_levels, np.array(densities)

    def detect_support_resistance_levels(self, price_levels, densities, current_price):
        """Detect support and resistance levels"""
        min_prominence = 0.05
        min_distance_between_levels = 0.003
        
        if price_levels is None or densities is None:
            return [], []
        
        if len(densities) > 0:
            densities = densities / np.max(densities)
        
        significant_levels = []
        significant_densities = []
        
        for i, (level, density) in enumerate(zip(price_levels, densities)):
            if density > min_prominence:
                too_close = False
                for existing_level in significant_levels:
                    if abs(level - existing_level) / current_price < min_distance_between_levels:
                        too_close = True
                        break
                
                if not too_close:
                    significant_levels.append(level)
                    significant_densities.append(density)
        
        if significant_levels:
            sorted_indices = np.argsort(significant_densities)[::-1]
            significant_levels = [significant_levels[i] for i in sorted_indices]
        
        support_levels = [level for level in significant_levels if level < current_price]
        resistance_levels = [level for level in significant_levels if level > current_price]
        
        return support_levels, resistance_levels

    def calculate_volatility(self, df, current_idx, period=20):
        """Calculate volatility metrics"""
        if current_idx < period:
            return 0.02
        
        recent_data = df.iloc[current_idx - period:current_idx]
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(24 * 12)
        
        return volatility

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Adds several technical indicators to the given DataFrame
        """
        logger.info("🔧 Populating indicators for OptimizedXGBoostStrategy...")
        
        # Calculate basic indicators
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_21'] = ta.RSI(dataframe, timeperiod=21)
        dataframe['macd'] = ta.MACD(dataframe)['macd']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Calculate volatility
        dataframe['volatility'] = dataframe['close'].pct_change().rolling(window=20).std() * np.sqrt(24 * 12)
        
        # Calculate price momentum
        dataframe['price_momentum_1'] = dataframe['close'].pct_change(1)
        dataframe['price_momentum_3'] = dataframe['close'].pct_change(3)
        dataframe['price_momentum_5'] = dataframe['close'].pct_change(5)
        
        # Calculate volume momentum (original algorithm method)
        dataframe['volume_momentum'] = 0.0
        for i in range(len(dataframe)):
            if i >= 5:
                volume_mean = dataframe.iloc[i-5:i]['volume'].mean()
                if volume_mean > 0:
                    dataframe.iloc[i, dataframe.columns.get_loc('volume_momentum')] = (
                        (dataframe.iloc[i]['volume'] - volume_mean) / volume_mean
                    )
        
        # Calculate volume ratio
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume'].rolling(window=20).mean()
        
        # Calculate normalized ATR
        dataframe['normalized_atr'] = dataframe['atr'] / dataframe['close']
        
        # Calculate price change
        dataframe['price_change'] = dataframe['close'].pct_change()
        
        # Calculate support and resistance levels using market profile
        dataframe['nearest_support'] = 0.0
        dataframe['nearest_resistance'] = 0.0
        
        # Calculate market profile-based support/resistance for each row
        for i in range(len(dataframe)):
            if i >= 200:  # Lookback period
                price_levels, densities = self.calculate_fast_market_profile(dataframe, i)
                support_levels, resistance_levels = self.detect_support_resistance_levels(
                    price_levels, densities, dataframe.iloc[i]['close']
                )
                
                nearest_support = max(support_levels) if support_levels else dataframe.iloc[i]['close'] * 0.95
                nearest_resistance = min(resistance_levels) if resistance_levels else dataframe.iloc[i]['close'] * 1.05
                
                dataframe.iloc[i, dataframe.columns.get_loc('nearest_support')] = nearest_support
                dataframe.iloc[i, dataframe.columns.get_loc('nearest_resistance')] = nearest_resistance
            else:
                dataframe.iloc[i, dataframe.columns.get_loc('nearest_support')] = dataframe.iloc[i]['close'] * 0.95
                dataframe.iloc[i, dataframe.columns.get_loc('nearest_resistance')] = dataframe.iloc[i]['close'] * 1.05
        
        # Calculate distances to support/resistance
        dataframe['distance_to_support'] = (dataframe['close'] - dataframe['nearest_support']) / dataframe['close']
        dataframe['distance_to_resistance'] = (dataframe['nearest_resistance'] - dataframe['close']) / dataframe['close']
        
        # Calculate higher high and lower low
        dataframe['higher_high'] = (dataframe['close'] > dataframe['high'].rolling(window=5).max().shift(1)).astype(int)
        dataframe['lower_low'] = (dataframe['close'] < dataframe['low'].rolling(window=5).min().shift(1)).astype(int)
        
        # Calculate breakouts
        dataframe['breakout_up'] = (dataframe['close'] > dataframe['nearest_resistance']).astype(int)
        dataframe['breakout_down'] = (dataframe['close'] < dataframe['nearest_support']).astype(int)
        
        # Calculate volatility ratio
        dataframe['volatility_ratio'] = dataframe['volatility'] / 0.02
        
        # Calculate price volatility
        dataframe['price_volatility'] = dataframe['close'].rolling(window=20).std() / dataframe['close'].rolling(window=20).mean()
        
        # Add time-based features
        if hasattr(dataframe.index, 'hour'):
            dataframe['hour'] = dataframe.index.hour
            dataframe['day_of_week'] = dataframe.index.dayofweek
        else:
            # If index is not datetime, use default values
            dataframe['hour'] = 12  # Default to noon
            dataframe['day_of_week'] = 1  # Default to Tuesday
        
        # Calculate XGBoost predictions if models are loaded
        if hasattr(self, 'models') and hasattr(self, 'scaler') and self.models and self.scaler:
            dataframe = self.calculate_xgboost_predictions(dataframe)
        
        logger.info(f"✅ Indicators populated for {len(dataframe)} candles")
        return dataframe

    def calculate_xgboost_predictions(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate XGBoost ensemble predictions"""
        try:
            if not hasattr(self, 'scaler') or not hasattr(self, 'models'):
                logger.warning("Models not loaded, skipping XGBoost predictions")
                dataframe['xgboost_prediction'] = 0
                dataframe['xgboost_confidence'] = 0.5
                return dataframe
            
            # Get feature columns (same as used in training)
            feature_columns = [
                'price_change', 'volume_ratio', 'normalized_atr',
                'nearest_support', 'nearest_resistance',
                'distance_to_support', 'distance_to_resistance',
                'price_momentum_1', 'price_momentum_3', 'price_momentum_5',
                'volume_momentum', 'rsi_14', 'rsi_21', 'macd',
                'higher_high', 'lower_low', 'breakout_up', 'breakout_down',
                'volatility_ratio', 'price_volatility',
                'hour', 'day_of_week',
                'volatility'
            ]
            
            # Check if all required columns exist
            missing_columns = [col for col in feature_columns if col not in dataframe.columns]
            if missing_columns:
                logger.warning(f"Missing columns for XGBoost: {missing_columns}")
                dataframe['xgboost_prediction'] = 0
                dataframe['xgboost_confidence'] = 0.5
                return dataframe
            
            # Prepare features
            feature_data = dataframe[feature_columns].fillna(0)
            feature_data = np.nan_to_num(feature_data.values, nan=0.0, posinf=0.0, neginf=0.0)
            feature_data = np.clip(feature_data, -1e10, 1e10)
            
            # Scale features
            X_scaled = self.scaler.transform(feature_data)
            
            # Make ensemble predictions
            all_predictions = []
            all_probabilities = []
            
            for name, model in self.models.items():
                pred = model.predict(X_scaled)
                prob = model.predict_proba(X_scaled)
                all_predictions.append(pred)
                all_probabilities.append(prob)
            
            # Calculate ensemble predictions using dynamic weights
            ensemble_predictions = np.zeros_like(all_predictions[0], dtype=float)
            ensemble_probabilities = np.zeros_like(all_probabilities[0], dtype=float)
            
            # Use dynamic weights from ensemble_weights
            for i, (name, pred, prob) in enumerate(zip(self.models.keys(), all_predictions, all_probabilities)):
                weight = self.ensemble_weights.get(name, 1.0/len(self.models))
                ensemble_predictions += weight * pred.astype(float)
                ensemble_probabilities += weight * prob
            
            final_predictions = np.round(ensemble_predictions).astype(int)
            confidence_scores = np.max(ensemble_probabilities, axis=1)
            
            dataframe['xgboost_prediction'] = final_predictions
            dataframe['xgboost_confidence'] = confidence_scores
            
            logger.info(f"✅ XGBoost predictions calculated for {len(dataframe)} candles")
            logger.info(f"📊 Prediction distribution: Hold={np.sum(final_predictions==0)}, Long={np.sum(final_predictions==1)}, Short={np.sum(final_predictions==2)}")
            logger.info(f"📊 Average confidence: {np.mean(confidence_scores):.3f}")
            
        except Exception as e:
            logger.error(f"Error calculating XGBoost predictions: {e}")
            dataframe['xgboost_prediction'] = 0
            dataframe['xgboost_confidence'] = 0.5
        
        return dataframe

    def create_target(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable for model training (same as original algorithm)
        """
        targets = []
        
        for i in range(len(dataframe)):
            if i < 2:
                targets.append(0)
                continue
            
            current_price = dataframe.iloc[i]['close']
            future_price_1 = dataframe.iloc[i+1]['close'] if i+1 < len(dataframe) else current_price
            future_price_2 = dataframe.iloc[i+2]['close'] if i+2 < len(dataframe) else future_price_1
            
            return_1 = (future_price_1 - current_price) / current_price
            return_2 = (future_price_2 - current_price) / current_price
            
            avg_return = (return_1 + return_2) / 2
            
            if avg_return > 0.015:  # 1.5% target (optimized)
                targets.append(1)  # Long
            elif avg_return < -0.015:  # 1.5% target (optimized)
                targets.append(2)  # Short
            else:
                targets.append(0)  # Hold
        
        dataframe['target'] = targets
        return dataframe

    def retrain_models(self, dataframe: pd.DataFrame) -> bool:
        """
        Retrain models with new data (optional feature for live adaptation)
        """
        try:
            logger.info("🔄 Retraining models with new data...")
            
            # Create features and targets
            df_features = self.create_target(dataframe)
            
            # Prepare training data
            feature_columns = [col for col in df_features.columns 
                             if col not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
            
            X = df_features[feature_columns].values
            y = df_features['target'].values
            
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = np.clip(X, -1e10, 1e10)
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            self.scaler = scaler
            
            # Train models
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            model_scores = {}
            for name, model in self.models.items():
                logger.info(f"🔄 Retraining {name}...")
                model.fit(X_train, y_train)
                
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                model_scores[name] = test_score
                
                logger.info(f"✅ {name} - Train: {train_score:.4f}, Test: {test_score:.4f}")
            
            # Calculate dynamic weights based on performance
            total_score = sum(model_scores.values())
            self.ensemble_weights = {name: score/total_score for name, score in model_scores.items()}
            
            logger.info(f"📊 New ensemble weights: {self.ensemble_weights}")
            
            # Save updated models and weights
            self.save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return False

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        # Initialize columns if they don't exist
        if 'xgboost_prediction' not in dataframe.columns:
            dataframe['xgboost_prediction'] = 0
        if 'xgboost_confidence' not in dataframe.columns:
            dataframe['xgboost_confidence'] = 0.5
        
        # Long entry conditions
        dataframe.loc[
            (
                (dataframe['xgboost_prediction'] == 1) &  # XGBoost predicts long
                (dataframe['xgboost_confidence'] >= self.min_confidence_for_trade.value) &
                (dataframe['volatility'] <= self.max_volatility_threshold.value) &
                (dataframe['volume_ratio'] > 1.0) &  # Above average volume
                (dataframe['rsi_14'] < 70) &  # Not overbought
                (dataframe['close'] > dataframe['close'].shift(1))  # Price momentum
            ),
            'enter_long'] = 1
        
        # Short entry conditions
        dataframe.loc[
            (
                (dataframe['xgboost_prediction'] == 2) &  # XGBoost predicts short
                (dataframe['xgboost_confidence'] >= self.min_confidence_for_trade.value) &
                (dataframe['volatility'] <= self.max_volatility_threshold.value) &
                (dataframe['volume_ratio'] > 1.0) &  # Above average volume
                (dataframe['rsi_14'] > 30) &  # Not oversold
                (dataframe['close'] < dataframe['close'].shift(1))  # Price momentum
            ),
            'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        # Initialize columns if they don't exist
        if 'xgboost_prediction' not in dataframe.columns:
            dataframe['xgboost_prediction'] = 0
        if 'xgboost_confidence' not in dataframe.columns:
            dataframe['xgboost_confidence'] = 0.5
        
        # Exit long positions
        dataframe.loc[
            (
                (dataframe['xgboost_prediction'] == 2) |  # XGBoost predicts short
                (dataframe['rsi_14'] > 80) |  # Overbought
                (dataframe['close'] < dataframe['nearest_support'])  # Break below support
            ),
            'exit_long'] = 1
        
        # Exit short positions
        dataframe.loc[
            (
                (dataframe['xgboost_prediction'] == 1) |  # XGBoost predicts long
                (dataframe['rsi_14'] < 20) |  # Oversold
                (dataframe['close'] > dataframe['nearest_resistance'])  # Break above resistance
            ),
            'exit_short'] = 1
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, after_fill: bool, **kwargs) -> float:
        """
        Custom stoploss logic using the optimized parameters
        """
        # Use the optimized stoploss value
        return self.stoploss

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        Custom exit logic for emergency exits
        """
        # Emergency exit if confidence drops significantly
        if hasattr(trade, 'enter_tag') and 'confidence' in str(trade.enter_tag):
            try:
                confidence = float(trade.enter_tag.split('_')[-1])
                if confidence < self.min_confidence_for_trade.value * 0.8:
                    return 'confidence_drop'
            except:
                pass
        
        # Emergency exit if volatility spikes
        if hasattr(self, 'dataprovider') and self.dataprovider:
            try:
                dataframe, _ = self.dataprovider.get_analyzed_dataframe(pair, self.timeframe)
                if len(dataframe) > 0:
                    current_volatility = dataframe['volatility'].iloc[-1]
                    if current_volatility > self.max_volatility_threshold.value * 1.5:
                        return 'volatility_spike'
            except:
                pass
        
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                          time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                          side: str, **kwargs) -> bool:
        """
        Confirm trade entry with additional checks
        """
        # Add confidence to entry tag
        if hasattr(self, 'dataprovider') and self.dataprovider:
            try:
                dataframe, _ = self.dataprovider.get_analyzed_dataframe(pair, self.timeframe)
                if len(dataframe) > 0 and 'xgboost_confidence' in dataframe.columns:
                    confidence = dataframe['xgboost_confidence'].iloc[-1]
                    if entry_tag:
                        entry_tag += f"_conf_{confidence:.3f}"
                    else:
                        entry_tag = f"conf_{confidence:.3f}"
            except:
                pass
        
        return True

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Custom leverage calculation based on confidence
        """
        if hasattr(self, 'dataprovider') and self.dataprovider:
            try:
                dataframe, _ = self.dataprovider.get_analyzed_dataframe(pair, self.timeframe)
                if len(dataframe) > 0 and 'xgboost_confidence' in dataframe.columns:
                    confidence = dataframe['xgboost_confidence'].iloc[-1]
                    volatility = dataframe['volatility'].iloc[-1]
                    
                    # Calculate dynamic leverage based on confidence
                    if confidence >= self.min_confidence_for_max_leverage.value:
                        leverage = self.max_leverage.value
                    elif confidence >= self.min_confidence_for_leverage.value:
                        leverage = 2.0 + ((confidence - self.min_confidence_for_leverage.value) / 
                                        (self.min_confidence_for_max_leverage.value - self.min_confidence_for_leverage.value)) * 3.0
                    else:
                        leverage = 1.0
                    
                    # Reduce leverage if volatility is high
                    if volatility > self.max_volatility_threshold.value:
                        leverage *= 0.5
                    
                    return min(leverage, self.max_leverage.value)
            except:
                pass
        
        return 1.0

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """
        Custom position sizing based on confidence
        """
        if hasattr(self, 'dataprovider') and self.dataprovider:
            try:
                dataframe, _ = self.dataprovider.get_analyzed_dataframe(pair, self.timeframe)
                if len(dataframe) > 0 and 'xgboost_confidence' in dataframe.columns:
                    confidence = dataframe['xgboost_confidence'].iloc[-1]
                    volatility = dataframe['volatility'].iloc[-1]
                    
                    # Calculate position size based on confidence
                    position_size = self.base_position_size.value + (confidence - 0.63) * 0.10
                    position_size = np.clip(position_size, self.base_position_size.value, self.max_position_size.value)
                    
                    # Reduce position size if volatility is high
                    if volatility > self.max_volatility_threshold.value * 0.8:
                        position_size *= 0.7
                    
                    # Calculate stake amount
                    stake_amount = proposed_stake * position_size
                    return min(stake_amount, max_stake)
            except:
                pass
        
        return proposed_stake
