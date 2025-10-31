#!/usr/bin/env python3

"""
OPTIMIZED HIGH-RETURN RISK-MANAGED TRADING SYSTEM
Parameters optimized for higher returns while maintaining safety

Original Performance: 47.13% return, 90.74% win rate
Optimized Target: 60-80% return with maintained safety

Key Optimizations:
1. Take-Profit: 1.5% (vs 1% original) - Better R:R ratio
2. Position Size: 30-35% (vs 25-30% original) - Slightly larger
3. Min Confidence: 63% (vs 65% original) - More opportunities
4. Max Leverage: 5x (vs 4x original) - Higher risk/reward
5. Stop-Loss: 1.8% (vs 2% original) - Tighter stops

All safety features maintained:
- MAE Monitoring: Active
- Portfolio DD Limit: 15%
- Daily Loss Limit: 5%
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class OptimizedHighReturnSystem:
    def __init__(self, max_leverage=5):
        """
        Initialize optimized high-return risk-managed trading system
        
        Args:
            max_leverage: Maximum leverage to use (5x optimized vs 4x original)
        """
        
        # ===== OPTIMIZED PARAMETERS =====
        self.initial_capital = 10000.0
        self.commission_rate = 0.0005  # 0.05%
        self.slippage_rate = 0.0002    # 0.02%
        self.min_trade_amount = 50.0
        
        # ===== OPTIMIZED POSITION SIZING (Increased) =====
        self.base_position_size = 0.30  # 30% base (vs 25% original)
        self.max_position_size = 0.35   # 35% maximum (vs 30% original)
        
        # ===== OPTIMIZED STOP-LOSS & TAKE-PROFIT =====
        self.stop_loss_pct = 0.018    # 1.8% (tighter than 2% original)
        self.take_profit_pct = 0.015  # 1.5% always (vs 1% original - better R:R)
        
        # ===== OPTIMIZED CONFIDENCE THRESHOLDS (More Aggressive) =====
        self.min_confidence_for_trade = 0.63  # 63% minimum (vs 65% original)
        self.min_confidence_for_leverage = 0.73  # 73% for any leverage (vs 75%)
        self.min_confidence_for_max_leverage = 0.88  # 88% for max leverage (vs 90%)
        
        # ===== OPTIMIZED LEVERAGE MANAGEMENT =====
        self.max_leverage = max_leverage  # 5x (vs 4x original)
        self.default_leverage = 1.0
        
        # ===== MAE MONITORING (CRITICAL FOR SURVIVAL) =====
        self.mae_monitoring_active = True
        self.mae_threshold_multiplier = 2.0  # Exit if unrealized loss > 2x stop-loss
        
        # ===== PORTFOLIO RISK CONTROLS (UNCHANGED - SAFETY FIRST) =====
        self.portfolio_max_drawdown_pct = 0.15  # 15% hard stop
        self.portfolio_drawdown_warning_pct = 0.10  # 10% warning - reduce sizing
        self.daily_loss_limit_pct = 0.05  # 5% daily max loss
        
        # ===== VOLATILITY FILTER (Slightly More Permissive) =====
        self.max_volatility_threshold = 0.022  # 2.2% (vs 2% original - more opportunities)
        
        # ===== KDE MARKET PROFILE PARAMETERS =====
        self.lookback_period = 200
        self.atr_period = 14
        self.kde_bandwidth_multiplier = 0.3
        self.time_decay_factor = 0.98
        self.min_prominence = 0.05
        self.min_distance_between_levels = 0.003
        
        # ===== TRACKING VARIABLES =====
        self.equity_peak = self.initial_capital
        self.daily_starting_equity = self.initial_capital
        self.current_equity = self.initial_capital
        self.current_drawdown_pct = 0
        self.daily_pnl = 0
        self.trading_halted = False
        self.halt_reason = None
        self.mae_exits_count = 0
        self.drawdown_halts_count = 0
        
        logger.info(f"🚀 Optimized High-Return System initialized (Max Leverage: {max_leverage}x)")
        logger.info(f"📊 Position Size: {self.base_position_size*100:.0f}%-{self.max_position_size*100:.0f}% (OPTIMIZED)")
        logger.info(f"📊 Stop-Loss: {self.stop_loss_pct*100:.2f}% | Take-Profit: {self.take_profit_pct*100:.2f}% (OPTIMIZED)")
        logger.info(f"📊 MAE Monitoring: {'Active' if self.mae_monitoring_active else 'Inactive'}")
        logger.info(f"📊 Portfolio Max DD: {self.portfolio_max_drawdown_pct*100:.0f}% (UNCHANGED)")
        logger.info(f"📊 Daily Loss Limit: {self.daily_loss_limit_pct*100:.0f}% (UNCHANGED)")
    
    # Copy all other methods from complete_high_return_system.py
    def load_binance_data(self, timeframe='5m'):
        """Load Binance data"""
        try:
            data_path = Path("data/binance/BTC_USDT-5m.json")
            if not data_path.exists():
                data_path = Path("../data/binance/BTC_USDT-5m.json")
            if not data_path.exists():
                data_path = Path("BTC_USDT-5m.json")
            
            if not data_path.exists():
                logger.error(f"Data file not found: {data_path}")
                return None
            
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            logger.info(f"✅ Data loaded: {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def calculate_time_weights(self, lookback_period):
        """Calculate exponential time decay weights"""
        weights = np.exp(np.linspace(-2, 0, lookback_period))
        return weights / weights.sum()

    def calculate_fast_market_profile(self, df, current_idx):
        """Calculate fast market profile using quantiles and pivot points"""
        if current_idx < self.lookback_period:
            return None, None
        
        recent_data = df.iloc[current_idx - self.lookback_period:current_idx]
        prices = recent_data['close'].values
        volumes = recent_data['volume'].values
        
        time_weights = self.calculate_time_weights(len(prices))
        
        high_low = recent_data['high'] - recent_data['low']
        high_close = np.abs(recent_data['high'] - recent_data['close'].shift(1))
        low_close = np.abs(recent_data['low'] - recent_data['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=self.atr_period).mean().iloc[-1]
        
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        price_levels = np.quantile(prices, quantiles)
        
        densities = []
        for level in price_levels:
            distance = np.abs(prices - level)
            bandwidth = atr * self.kde_bandwidth_multiplier
            
            weights = np.exp(-0.5 * (distance / bandwidth) ** 2) * volumes * time_weights
            density = np.sum(weights)
            densities.append(density)
        
        return price_levels, np.array(densities)

    def detect_support_resistance_levels(self, price_levels, densities, current_price):
        """Detect support and resistance levels"""
        if price_levels is None or densities is None:
            return [], []
        
        if len(densities) > 0:
            densities = densities / np.max(densities)
        
        significant_levels = []
        significant_densities = []
        
        for i, (level, density) in enumerate(zip(price_levels, densities)):
            if density > self.min_prominence:
                too_close = False
                for existing_level in significant_levels:
                    if abs(level - existing_level) / current_price < self.min_distance_between_levels:
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
    
    def create_features(self, df):
        """Create features for the optimized high-return system"""
        logger.info("🔧 Creating features for Optimized High-Return System...")
        
        features = []
        step_size = 6
        
        for i in range(self.lookback_period, len(df), step_size):
            if i % 10000 == 0:
                logger.info(f"Processing record {i}/{len(df)}")
            
            price_levels, densities = self.calculate_fast_market_profile(df, i)
            support_levels, resistance_levels = self.detect_support_resistance_levels(
                price_levels, densities, df.iloc[i]['close']
            )
            
            volatility = self.calculate_volatility(df, i)
            
            current_price = df.iloc[i]['close']
            price_change = (current_price - df.iloc[i-1]['close']) / df.iloc[i-1]['close']
            volume_ratio = df.iloc[i]['volume'] / df.iloc[i-20:i]['volume'].mean() if i >= 20 else 1.0
            
            high_low = df.iloc[i-14:i]['high'] - df.iloc[i-14:i]['low']
            high_close = np.abs(df.iloc[i-14:i]['high'] - df.iloc[i-14:i]['close'].shift(1))
            low_close = np.abs(df.iloc[i-14:i]['low'] - df.iloc[i-14:i]['close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.mean()
            normalized_atr = atr / current_price
            
            nearest_support = max(support_levels) if support_levels else current_price * 0.95
            nearest_resistance = min(resistance_levels) if resistance_levels else current_price * 1.05
            
            distance_to_support = (current_price - nearest_support) / current_price
            distance_to_resistance = (nearest_resistance - current_price) / current_price
            
            price_momentum_1 = (current_price - df.iloc[i-1]['close']) / df.iloc[i-1]['close']
            price_momentum_3 = (current_price - df.iloc[i-3]['close']) / df.iloc[i-3]['close']
            price_momentum_5 = (current_price - df.iloc[i-5]['close']) / df.iloc[i-5]['close']
            
            volume_momentum = (df.iloc[i]['volume'] - df.iloc[i-5:i]['volume'].mean()) / df.iloc[i-5:i]['volume'].mean()
            
            rsi_14 = self._calculate_rsi(df.iloc[i-14:i+1]['close'].values)
            rsi_21 = self._calculate_rsi(df.iloc[i-21:i+1]['close'].values)
            macd = self._calculate_macd(df.iloc[i-26:i+1]['close'].values)
            
            higher_high = current_price > df.iloc[i-5:i]['high'].max()
            lower_low = current_price < df.iloc[i-5:i]['low'].min()
            
            breakout_up = current_price > nearest_resistance
            breakout_down = current_price < nearest_support
            
            volatility_ratio = volatility / 0.02
            price_volatility = df.iloc[i-20:i]['close'].std() / df.iloc[i-20:i]['close'].mean()
            
            hour = df.iloc[i]['timestamp'].hour
            day_of_week = df.iloc[i]['timestamp'].weekday()
            
            feature_vector = [
                price_change, volume_ratio, normalized_atr,
                nearest_support, nearest_resistance,
                distance_to_support, distance_to_resistance,
                price_momentum_1, price_momentum_3, price_momentum_5,
                volume_momentum, rsi_14, rsi_21, macd,
                higher_high, lower_low, breakout_up, breakout_down,
                volatility_ratio, price_volatility,
                hour, day_of_week,
                volatility
            ]
            
            features.append(feature_vector)
        
        feature_names = [
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
        
        df_features = pd.DataFrame(features, columns=feature_names)
        
        timestamps = df.iloc[self.lookback_period::step_size]['timestamp'].values
        df_features['timestamp'] = timestamps[:len(df_features)]
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_features[col] = df.iloc[self.lookback_period::step_size][col].values[:len(df_features)]
        
        df_features['target'] = self._create_target(df_features)
        df_features = df_features.fillna(method='ffill').fillna(0)
        
        logger.info(f"✅ Features created: {len(df_features)} records")
        
        return df_features.dropna()

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        if avg_losses == 0:
            return 100
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices):
        """Calculate MACD"""
        if len(prices) < 26:
            return 0
        
        ema_12 = pd.Series(prices).ewm(span=12).mean().iloc[-1]
        ema_26 = pd.Series(prices).ewm(span=26).mean().iloc[-1]
        macd = ema_12 - ema_26
        return macd

    def _create_target(self, df):
        """Create target variable"""
        targets = []
        
        for i in range(len(df)):
            if i < 2:
                targets.append(0)
                continue
            
            current_price = df.iloc[i]['close']
            future_price_1 = df.iloc[i+1]['close'] if i+1 < len(df) else current_price
            future_price_2 = df.iloc[i+2]['close'] if i+2 < len(df) else future_price_1
            
            return_1 = (future_price_1 - current_price) / current_price
            return_2 = (future_price_2 - current_price) / current_price
            
            avg_return = (return_1 + return_2) / 2
            
            if avg_return > 0.015:  # 1.5% target (optimized)
                targets.append(1)  # Long
            elif avg_return < -0.015:  # 1.5% target (optimized)
                targets.append(2)  # Short
            else:
                targets.append(0)  # Hold
        
        return targets

    def create_models(self):
        """Create ensemble models"""
        logger.info("🤖 Creating ensemble XGBoost models...")
        
        self.models = {
            'xgboost_performance': xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'xgboost_aggressive': xgb.XGBClassifier(
                n_estimators=150, max_depth=7, learning_rate=0.15,
                subsample=0.85, colsample_bytree=0.85, random_state=42
            ),
            'xgboost_balanced': xgb.XGBClassifier(
                n_estimators=100, max_depth=8, learning_rate=0.2,
                subsample=0.9, colsample_bytree=0.9, random_state=42
            )
        }
        
        logger.info("✅ Created 3 ensemble models")

    def train_models(self, df):
        """Train models"""
        logger.info("🎓 Training ensemble models...")
        
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
        X = df[feature_columns].values
        y = df['target'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -1e10, 1e10)
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Data split: {len(X_train)} training, {len(X_test)} testing")
        
        model_scores = {}
        for name, model in self.models.items():
            logger.info(f"🔄 Training {name}...")
            model.fit(X_train, y_train)
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            model_scores[name] = test_score
            
            logger.info(f"✅ {name} - Train: {train_score:.4f}, Test: {test_score:.4f}")
        
        total_score = sum(model_scores.values())
        self.ensemble_weights = {name: score/total_score for name, score in model_scores.items()}
        
        logger.info(f"📊 Ensemble weights: {self.ensemble_weights}")
        
        return X_test, y_test
    
    def save_models(self):
        """Save trained models and scaler"""
        try:
            models_path = Path("user_data/models")
            models_path.mkdir(exist_ok=True)
            
            # Save scaler
            if hasattr(self, 'scaler') and self.scaler:
                import joblib
                joblib.dump(self.scaler, models_path / "scaler.pkl")
                logger.info("✅ Scaler saved")
            
            # Save models
            if hasattr(self, 'models') and self.models:
                import joblib
                for name, model in self.models.items():
                    joblib.dump(model, models_path / f"{name}.pkl")
                logger.info("✅ Models saved")
            
            # Save feature columns
            if hasattr(self, 'df_with_features') and self.df_with_features is not None:
                feature_columns = [col for col in self.df_with_features.columns 
                                 if col not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
                with open(models_path / "feature_columns.json", 'w') as f:
                    json.dump(feature_columns, f)
                logger.info("✅ Feature columns saved")
            
            # Save ensemble weights (same logic as used in predictions)
            if hasattr(self, 'ensemble_weights') and self.ensemble_weights:
                with open(models_path / "ensemble_weights.json", 'w') as f:
                    json.dump(self.ensemble_weights, f, indent=2)
                logger.info("✅ Ensemble weights saved")
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def _ensemble_predict(self, X_scaled):
        """Make ensemble predictions"""
        all_predictions = []
        all_probabilities = []
        
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            prob = model.predict_proba(X_scaled)
            all_predictions.append(pred)
            all_probabilities.append(prob)
        
        ensemble_predictions = np.zeros_like(all_predictions[0], dtype=float)
        ensemble_probabilities = np.zeros_like(all_probabilities[0], dtype=float)
        
        for i, (name, pred, prob) in enumerate(zip(self.models.keys(), all_predictions, all_probabilities)):
            weight = self.ensemble_weights[name]
            ensemble_predictions += weight * pred.astype(float)
            ensemble_probabilities += weight * prob
        
        final_predictions = np.round(ensemble_predictions).astype(int)
        confidence_scores = np.max(ensemble_probabilities, axis=1)
        
        return final_predictions, confidence_scores
    
    def update_equity(self, new_equity):
        """Update current equity and track peak"""
        self.current_equity = new_equity
        
        if new_equity > self.equity_peak:
            self.equity_peak = new_equity
        
        self.current_drawdown_pct = (self.equity_peak - new_equity) / self.equity_peak
    
    def check_portfolio_risk_limits(self):
        """Check if portfolio risk limits are breached"""
        if self.current_drawdown_pct >= self.portfolio_max_drawdown_pct:
            reason = f"Portfolio DD {self.current_drawdown_pct*100:.2f}% >= {self.portfolio_max_drawdown_pct*100:.0f}%"
            logger.warning(f"🛑 {reason}")
            self.drawdown_halts_count += 1
            return True, reason
        
        daily_loss = (self.current_equity - self.daily_starting_equity) / self.daily_starting_equity
        if daily_loss <= -self.daily_loss_limit_pct:
            reason = f"Daily loss {abs(daily_loss)*100:.2f}% >= {self.daily_loss_limit_pct*100:.0f}%"
            logger.warning(f"🛑 {reason}")
            return True, reason
        
        return False, None
    
    def calculate_dynamic_leverage(self, confidence, volatility):
        """Calculate leverage based on confidence and volatility - OPTIMIZED"""
        if confidence < self.min_confidence_for_leverage:
            return 1.0
        
        if volatility > self.max_volatility_threshold:
            return 1.0
        
        # OPTIMIZED: More aggressive leverage scaling
        if confidence >= self.min_confidence_for_max_leverage:
            leverage_range = self.max_leverage - 3
            confidence_factor = (confidence - self.min_confidence_for_max_leverage) / (1.0 - self.min_confidence_for_max_leverage)
            leverage = 3.0 + (leverage_range * confidence_factor)
        elif confidence >= 0.75:
            leverage = 2.0 + ((confidence - 0.75) / (self.min_confidence_for_max_leverage - 0.75))
        else:
            leverage = 1.5 + ((confidence - self.min_confidence_for_leverage) / (0.75 - self.min_confidence_for_leverage))
        
        if self.current_drawdown_pct >= self.portfolio_drawdown_warning_pct:
            leverage *= 0.5
            logger.info(f"⚠️  Leverage reduced due to drawdown warning")
        
        return min(leverage, self.max_leverage)
    
    def calculate_position_size(self, confidence, volatility):
        """Calculate position size based on confidence - OPTIMIZED"""
        # OPTIMIZED: Larger base position size
        position_size = self.base_position_size + (confidence - 0.63) * 0.10  # Adjusted for new threshold
        position_size = np.clip(position_size, self.base_position_size, self.max_position_size)
        
        if self.current_drawdown_pct >= self.portfolio_drawdown_warning_pct:
            position_size *= 0.5
        
        if volatility > self.max_volatility_threshold * 0.8:
            position_size *= 0.7
        
        return position_size
    
    def check_mae_threshold(self, unrealized_loss, position_value, leverage):
        """Check if MAE exceeds emergency threshold"""
        if not self.mae_monitoring_active:
            return False
        
        expected_max_loss = position_value * self.stop_loss_pct
        mae_threshold = expected_max_loss * self.mae_threshold_multiplier
        
        if abs(unrealized_loss) >= mae_threshold:
            logger.warning(f"🚨 MAE threshold breached!")
            self.mae_exits_count += 1
            return True
        
        return False
    
    def should_enter_trade(self, confidence, volatility):
        """Determine if should enter a trade"""
        if self.trading_halted:
            return False, 0, 0, f"Trading halted: {self.halt_reason}"
        
        should_halt, halt_reason = self.check_portfolio_risk_limits()
        if should_halt:
            self.trading_halted = True
            self.halt_reason = halt_reason
            return False, 0, 0, halt_reason
        
        if confidence < self.min_confidence_for_trade:
            return False, 0, 0, f"Confidence {confidence:.2%} < {self.min_confidence_for_trade:.0%}"
        
        if volatility > self.max_volatility_threshold:
            return False, 0, 0, f"Volatility {volatility:.2%} > {self.max_volatility_threshold:.0%}"
        
        leverage = self.calculate_dynamic_leverage(confidence, volatility)
        position_size = self.calculate_position_size(confidence, volatility)
        
        return True, leverage, position_size, "All checks passed"
    
    def run_backtest(self, use_sample=False):
        """Run complete backtest"""
        logger.info("🚀 Starting Optimized High-Return System Backtest...")
        
        df = self.load_binance_data()
        if df is None:
            return None
        
        logger.info(f"📊 Loaded {len(df)} records")
        
        if use_sample:
            df = df.tail(50000)
            logger.info(f"📊 Using sample: {len(df)} records")
        
        df_features = self.create_features(df)
        if df_features is None or len(df_features) == 0:
            logger.error("Failed to create features")
            return None
        
        self.df_with_features = df_features
        
        self.create_models()
        X_test, y_test = self.train_models(df_features)
        
        logger.info("📈 Running optimized high-return backtest...")
        start_time = datetime.now()
        
        performance, portfolio = self._run_trading_simulation(df_features)
        
        end_time = datetime.now()
        logger.info(f"✅ Backtest completed in {(end_time - start_time).total_seconds():.2f} seconds")
        
        self._generate_report(performance, portfolio)
        
        logger.info("🎉 Optimized backtest completed!")
        return performance, portfolio

    def _run_trading_simulation(self, df):
        """Run trading simulation with optimized parameters"""
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
        X = df[feature_columns].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -1e10, 1e10)
        X_scaled = self.scaler.transform(X)
        
        predictions, confidence_scores = self._ensemble_predict(X_scaled)
        
        portfolio = {
            'cash': self.initial_capital,
            'position': 0,
            'position_size': 0,
            'entry_price': 0,
            'entry_time': None,
            'leverage': 1.0,
            'trades': [],
            'equity_curve': []
        }
        
        for i in range(len(df)):
            current_time = df.iloc[i]['timestamp']
            current_price = df.iloc[i]['close']
            prediction = predictions[i]
            confidence = confidence_scores[i]
            volatility = df.iloc[i]['volatility']
            
            if portfolio['position'] == 0:
                should_enter, leverage, position_size, reason = self.should_enter_trade(confidence, volatility)
                
                if should_enter and prediction in [1, 2]:
                    if prediction == 1:
                        shares = (portfolio['cash'] * position_size * leverage) / current_price
                        cost = shares * current_price
                        commission = cost * self.commission_rate
                        
                        portfolio['position'] = shares
                        portfolio['position_size'] = position_size
                        portfolio['entry_price'] = current_price
                        portfolio['entry_time'] = current_time
                        portfolio['leverage'] = leverage
                        portfolio['cash'] -= (cost + commission)
                        
                        portfolio['trades'].append({
                            'timestamp': current_time.timestamp(),
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'confidence': confidence,
                            'position_size': position_size,
                            'leverage': leverage,
                            'commission': commission
                        })
                    
                    elif prediction == 2:
                        shares = (portfolio['cash'] * position_size * leverage) / current_price
                        cost = shares * current_price
                        commission = cost * self.commission_rate
                        
                        portfolio['position'] = -shares
                        portfolio['position_size'] = position_size
                        portfolio['entry_price'] = current_price
                        portfolio['entry_time'] = current_time
                        portfolio['leverage'] = leverage
                        portfolio['cash'] += (cost - commission)
                        
                        portfolio['trades'].append({
                            'timestamp': current_time.timestamp(),
                            'action': 'SELL',
                            'price': current_price,
                            'shares': shares,
                            'confidence': confidence,
                            'position_size': position_size,
                            'leverage': leverage,
                            'commission': commission
                        })
            
            if portfolio['position'] != 0:
                stop_loss_price = portfolio['entry_price'] * (1 - self.stop_loss_pct if portfolio['position'] > 0 else 1 + self.stop_loss_pct)
                take_profit_price = portfolio['entry_price'] * (1 + self.take_profit_pct if portfolio['position'] > 0 else 1 - self.take_profit_pct)
                
                should_close = False
                close_reason = ""
                
                if portfolio['position'] > 0:
                    if current_price <= stop_loss_price:
                        should_close = True
                        close_reason = "stop_loss"
                    elif current_price >= take_profit_price:
                        should_close = True
                        close_reason = "take_profit"
                else:
                    if current_price >= stop_loss_price:
                        should_close = True
                        close_reason = "stop_loss"
                    elif current_price <= take_profit_price:
                        should_close = True
                        close_reason = "take_profit"
                
                if not should_close:
                    position_value = abs(portfolio['position']) * current_price
                    if portfolio['position'] > 0:
                        unrealized_loss = (current_price - portfolio['entry_price']) * abs(portfolio['position'])
                    else:
                        unrealized_loss = (portfolio['entry_price'] - current_price) * abs(portfolio['position'])
                    
                    if self.check_mae_threshold(unrealized_loss, position_value, portfolio['leverage']):
                        should_close = True
                        close_reason = "mae_emergency"
                
                if should_close:
                    shares = abs(portfolio['position'])
                    proceeds = shares * current_price
                    commission = proceeds * self.commission_rate
                    
                    if portfolio['position'] > 0:
                        portfolio['cash'] += (proceeds - commission)
                    else:
                        portfolio['cash'] -= (proceeds + commission)
                    
                    portfolio['trades'].append({
                        'timestamp': current_time.timestamp(),
                        'action': 'CLOSE',
                        'price': current_price,
                        'shares': shares,
                        'reason': close_reason,
                        'commission': commission
                    })
                    
                    portfolio['position'] = 0
                    portfolio['position_size'] = 0
                    portfolio['entry_price'] = 0
                    portfolio['entry_time'] = None
                    portfolio['leverage'] = 1.0
            
            current_equity = portfolio['cash'] + (portfolio['position'] * current_price if portfolio['position'] > 0 else 0) + (portfolio['position'] * current_price if portfolio['position'] < 0 else 0)
            portfolio['equity_curve'].append({
                'timestamp': current_time,
                'value': current_equity
            })
            self.update_equity(current_equity)
        
        if portfolio['position'] != 0:
            self._close_position(portfolio, df.iloc[-1]['close'], df.iloc[-1]['timestamp'], "end_of_data")
        
        performance = self._calculate_performance(portfolio)
        
        return performance, portfolio

    def _close_position(self, portfolio, price, time, reason):
        """Close position"""
        if portfolio['position'] == 0:
            return
        
        shares = abs(portfolio['position'])
        proceeds = shares * price
        commission = proceeds * self.commission_rate
        
        if portfolio['position'] > 0:
            portfolio['cash'] += (proceeds - commission)
        else:
            portfolio['cash'] -= (proceeds + commission)
        
        portfolio['trades'].append({
            'timestamp': time.timestamp(),
            'action': 'CLOSE',
            'price': price,
            'shares': shares,
            'reason': reason,
            'commission': commission
        })
        
        portfolio['position'] = 0
    
    def _calculate_performance(self, portfolio):
        """Calculate performance metrics"""
        final_value = portfolio['cash']
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        trades = portfolio['trades']
        completed_trades = len([t for t in trades if t['action'] == 'CLOSE'])
        
        if completed_trades == 0:
            return {
                'total_return': total_return,
                'win_rate': 0,
                'total_trades': len(trades),
                'completed_trades': completed_trades,
                'winning_trades': 0,
                'final_value': final_value
            }
        
        trade_pnls = []
        equity_values = [self.initial_capital]
        
        for trade in trades:
            if trade['action'] == 'CLOSE':
                entry_trade = None
                for i in range(len(trades) - 1, -1, -1):
                    if trades[i]['action'] in ['BUY', 'SELL'] and trades[i]['timestamp'] < trade['timestamp']:
                        entry_trade = trades[i]
                        break
                
                if entry_trade:
                    if entry_trade['action'] == 'BUY':
                        pnl = trade['shares'] * (trade['price'] - entry_trade['price'])
                    else:
                        pnl = trade['shares'] * (entry_trade['price'] - trade['price'])
                    
                    trade_pnls.append(pnl)
                    equity_values.append(equity_values[-1] + pnl)
        
        winning_trades = len([pnl for pnl in trade_pnls if pnl > 0])
        win_rate = (winning_trades / completed_trades) * 100 if completed_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'completed_trades': completed_trades,
            'winning_trades': winning_trades,
            'final_value': final_value,
            'equity_curve': portfolio['equity_curve']
        }

    def get_algorithm_info(self):
        """Get algorithm information for multi-algorithm system"""
        return {
            'name': 'Optimized High-Return System',
            'version': '2.0.0',
            'description': 'Advanced XGBoost ensemble with optimized risk management',
            'is_test_mode': False,
            'features': [
                'rsi', 'macd', 'atr', 'market_profile', 'volatility',
                'momentum', 'volume_profile', 'price_action'
            ],
            'parameters': {
                'max_leverage': self.max_leverage,
                'base_position_size': self.base_position_size,
                'max_position_size': self.max_position_size,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'min_confidence': self.min_confidence_for_trade,
                'mae_monitoring': self.mae_monitoring_active,
                'max_drawdown_pct': self.portfolio_max_drawdown_pct,
                'daily_loss_limit_pct': self.daily_loss_limit_pct
            }
        }

    def _generate_report(self, performance, portfolio):
        """Generate performance report"""
        logger.info("🏆 OPTIMIZED HIGH-RETURN RISK-MANAGED SYSTEM RESULTS:")
        logger.info("=" * 80)
        logger.info(f"Total Return: {performance['total_return']:.2f}%")
        logger.info(f"Win Rate: {performance['win_rate']:.2f}%")
        logger.info(f"Total Trades: {performance['total_trades']}")
        logger.info(f"Completed Trades: {performance['completed_trades']}")
        logger.info(f"Winning Trades: {performance['winning_trades']}")
        logger.info(f"Final Value: ${performance['final_value']:.2f}")
        logger.info("=" * 80)


def main():
    """Main function"""
    logger.info("🚀 Starting Optimized High-Return Risk-Managed Trading System")
    logger.info("📈 Parameters optimized for higher returns")
    
    # Initialize with 5x leverage (optimized)
    system = OptimizedHighReturnSystem(max_leverage=5)
    
    # Run backtest with full dataset for complete analysis
    result = system.run_backtest(use_sample=False)
    
    if result:
        logger.info("🎉 Optimized system test completed successfully!")
    else:
        logger.error("❌ Optimized system test failed!")


if __name__ == "__main__":
    main()
