#@version = 1
# Binance Futures Backtest Strategy - Ultra Simplified

LONG_CONFIDENCE_THRESHOLD = 0.42
LONG_STOP_LOSS_PCT = 0.006
LONG_TAKE_PROFIT_BASE = 0.005  # REDUCED from 0.02 (2%) to 0.5% for live trading
LONG_TAKE_PROFIT_MAX = 0.010  # REDUCED from 0.035 (3.5%) to 1.0% for live trading
BASE_POSITION_SIZE = 0.35
MAX_POSITION_SIZE = 0.7
STEP_SIZE = 6
RSI_PERIOD_1 = 14
RSI_PERIOD_2 = 21
MACD_FAST = 12
MACD_SLOW = 26
ATR_PERIOD = 14
VOLATILITY_PERIOD = 20
MOMENTUM_PERIOD = 5

last_evaluated_bar_index = 0
position_active = False
position_entry_price = 0.0
position_stop_loss = 0.0
position_take_profit = 0.0
position_quantity = 0.0

def get_close(bar):
    idx = 4
    return bar[idx]

def get_high(bar):
    idx = 2
    return bar[idx]

def get_low(bar):
    idx = 3
    return bar[idx]

def get_open(bar):
    idx = 1
    return bar[idx]

def get_volume(bar):
    idx = 5
    return bar[idx]

def calculate_rsi(close_prices, period):
    n = len(close_prices)
    if n < period + 1:
        return None
    
    changes = []
    for i in range(1, n):
        prev_idx = i - 1
        prev = close_prices[prev_idx]
        curr = close_prices[i]
        change = curr - prev
        changes.append(change)
    
    gains = []
    losses = []
    for i in range(len(changes)):
        change = changes[i]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-change)
    
    start_idx = len(gains) - period
    gain_sum = 0.0
    loss_sum = 0.0
    for i in range(start_idx, len(gains)):
        gain_sum = gain_sum + gains[i]
        loss_sum = loss_sum + losses[i]
    
    avg_gain = gain_sum / period
    avg_loss = loss_sum / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_ema(prices, period):
    n = len(prices)
    if n < period:
        return None
    
    multiplier = 2.0 / (period + 1.0)
    ema_sum = 0.0
    for i in range(period):
        ema_sum = ema_sum + prices[i]
    ema = ema_sum / period
    
    for i in range(period, n):
        price = prices[i]
        ema = (price - ema) * multiplier + ema
    
    return ema

def calculate_momentum(close_prices, period):
    n = len(close_prices)
    if n < period + 1:
        return None
    
    last_idx = n - 1
    current = close_prices[last_idx]
    past_idx = n - 1 - period
    past = close_prices[past_idx]
    momentum = (current - past) / past
    return momentum

def calculate_atr(highs, lows, closes, period):
    n = len(highs)
    if n < period + 1:
        return None
    
    true_ranges = []
    for i in range(1, n):
        high = highs[i]
        low = lows[i]
        prev_idx = i - 1
        prev_close = closes[prev_idx]
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        if tr1 > tr2:
            if tr1 > tr3:
                tr = tr1
            else:
                tr = tr3
        else:
            if tr2 > tr3:
                tr = tr2
            else:
                tr = tr3
        true_ranges.append(tr)
    
    if len(true_ranges) < period:
        return None
    
    start_idx = len(true_ranges) - period
    atr_sum = 0.0
    for i in range(start_idx, len(true_ranges)):
        atr_sum = atr_sum + true_ranges[i]
    
    atr = atr_sum / period
    return atr

def calculate_volatility(close_prices, period):
    n = len(close_prices)
    if n < period + 1:
        return None
    
    returns = []
    for i in range(1, n):
        prev_idx = i - 1
        prev = close_prices[prev_idx]
        curr = close_prices[i]
        ret = (curr - prev) / prev
        returns.append(ret)
    
    start_idx = len(returns) - period
    mean_sum = 0.0
    for i in range(start_idx, len(returns)):
        mean_sum = mean_sum + returns[i]
    mean = mean_sum / period
    
    variance_sum = 0.0
    for i in range(start_idx, len(returns)):
        diff = returns[i] - mean
        variance_sum = variance_sum + diff * diff
    
    variance = variance_sum / period
    std_dev = variance ** 0.5
    annualized = std_dev * 17.088
    return annualized

def calculate_confidence(bars):
    n = len(bars)
    if n < 50:
        return 0.0
    
    closes = []
    highs = []
    lows = []
    volumes = []
    
    for i in range(n):
        bar = bars[i]
        closes.append(get_close(bar))
        highs.append(get_high(bar))
        lows.append(get_low(bar))
        volumes.append(get_volume(bar))
    
    last_idx = n - 1
    current_bar = bars[last_idx]
    current_close = get_close(current_bar)
    current_high = get_high(current_bar)
    current_low = get_low(current_bar)
    current_volume = get_volume(current_bar)
    
    rsi14 = calculate_rsi(closes, RSI_PERIOD_1)
    rsi21 = calculate_rsi(closes, RSI_PERIOD_2)
    ema_fast = calculate_ema(closes, MACD_FAST)
    ema_slow = calculate_ema(closes, MACD_SLOW)
    atr = calculate_atr(highs, lows, closes, ATR_PERIOD)
    volatility = calculate_volatility(closes, VOLATILITY_PERIOD)
    
    if n >= 2:
        last_idx = n - 1
        prev_idx = n - 2
        curr_close = closes[last_idx]
        prev_close = closes[prev_idx]
        momentum1 = (curr_close - prev_close) / prev_close
    else:
        momentum1 = 0.0
    
    momentum5 = calculate_momentum(closes, MOMENTUM_PERIOD)
    
    if atr == None:
        normalized_atr = 0.0
    else:
        if current_close > 0:
            normalized_atr = atr / current_close
        else:
            normalized_atr = 0.0
    
    vol_start = n - 20
    if vol_start < 0:
        vol_start = 0
    
    avg_volume_sum = 0.0
    vol_count = 0
    for i in range(vol_start, n):
        avg_volume_sum = avg_volume_sum + volumes[i]
        vol_count = vol_count + 1
    
    if vol_count > 0:
        avg_volume = avg_volume_sum / vol_count
    else:
        avg_volume = 1.0
    
    if avg_volume > 0:
        volume_ratio = current_volume / avg_volume
    else:
        volume_ratio = 1.0
    
    macd_val = 0.0
    if ema_fast != None and ema_slow != None:
        macd_val = ema_fast - ema_slow
    
    higher_high = False
    if n >= 6:
        recent_highs = []
        start_high = n - 6
        end_high = n - 1
        for i in range(start_high, end_high):
            recent_highs.append(highs[i])
        max_high = recent_highs[0]
        for i in range(1, len(recent_highs)):
            if recent_highs[i] > max_high:
                max_high = recent_highs[i]
        if current_high > max_high:
            higher_high = True
    
    lower_low = False
    if n >= 6:
        recent_lows = []
        start_low = n - 6
        end_low = n - 1
        for i in range(start_low, end_low):
            recent_lows.append(lows[i])
        min_low = recent_lows[0]
        for i in range(1, len(recent_lows)):
            if recent_lows[i] < min_low:
                min_low = recent_lows[i]
        if current_low < min_low:
            lower_low = True
    
    confidence = 0.5
    
    if rsi14 != None:
        if rsi14 < 30:
            confidence = confidence + 0.1
        if rsi14 > 70:
            confidence = confidence - 0.15
        if rsi14 >= 40 and rsi14 <= 60:
            confidence = confidence + 0.05
    
    if rsi21 != None:
        if rsi21 < 30:
            confidence = confidence + 0.08
        if rsi21 > 70:
            confidence = confidence - 0.12
    
    if macd_val > 0:
        confidence = confidence + 0.08
    
    if momentum1 > 0:
        confidence = confidence + 0.06
    
    if momentum5 != None and momentum5 > 0.01:
        confidence = confidence + 0.08
    
    if volume_ratio > 1.2:
        confidence = confidence + 0.05
    
    if higher_high == True:
        confidence = confidence + 0.05
    
    if lower_low == True:
        confidence = confidence - 0.08
    
    if volatility != None and volatility > 0.3:
        confidence = confidence - 0.1
    
    if normalized_atr > 0.02:
        confidence = confidence - 0.05
    
    if confidence < 0:
        confidence = 0.0
    if confidence > 0:
        if confidence > 1:
            confidence = 1.0
    
    return confidence

def calculate_dynamic_take_profit(confidence):
    range_tp = LONG_TAKE_PROFIT_MAX - LONG_TAKE_PROFIT_BASE
    dynamic_tp = LONG_TAKE_PROFIT_BASE + (range_tp * confidence)
    return dynamic_tp

def calculate_position_size(confidence, available_capital):
    range_size = MAX_POSITION_SIZE - BASE_POSITION_SIZE
    position_size_pct = BASE_POSITION_SIZE + (range_size * confidence)
    position_size = available_capital * position_size_pct
    return position_size

def should_allow_long_signal(confidence):
    if confidence < 0.15:
        return False
    return True

def check_position_management(current_bar):
    global position_active
    global position_entry_price
    global position_stop_loss
    global position_take_profit
    global position_quantity
    
    if position_active == False:
        return
    
    current_high = get_high(current_bar)
    current_low = get_low(current_bar)
    
    should_close = False
    close_reason = 0
    
    if current_high >= position_take_profit:
        should_close = True
        close_reason = 1
    
    if current_low <= position_stop_loss:
        should_close = True
        close_reason = 2
    
    if should_close == True:
        if close_reason == 1:
            exit_price = position_take_profit
        else:
            exit_price = position_stop_loss
        
        pnl = (exit_price - position_entry_price) * position_quantity
        pnl_percent = ((exit_price - position_entry_price) / position_entry_price) * 100.0
        
        print("CLOSE")
        print(exit_price)
        print(pnl)
        
        position_active = False
        position_entry_price = 0.0
        position_stop_loss = 0.0
        position_take_profit = 0.0
        position_quantity = 0.0

def on_bar(bars):
    global last_evaluated_bar_index
    global position_active
    global position_entry_price
    global position_stop_loss
    global position_take_profit
    global position_quantity
    
    if len(bars) < 50:
        return
    
    last_idx = len(bars) - 1
    current_bar = bars[last_idx]
    current_price = get_close(current_bar)
    bar_index = len(bars) - 1
    
    if bar_index % STEP_SIZE != 0 and last_evaluated_bar_index != 0:
        check_position_management(current_bar)
        return
    
    last_evaluated_bar_index = bar_index
    
    if position_active == True:
        check_position_management(current_bar)
        if position_active == False:
            return
    
    if position_active == True:
        return
    
    confidence = calculate_confidence(bars)
    
    if confidence < LONG_CONFIDENCE_THRESHOLD:
        return
    
    if should_allow_long_signal(confidence) == False:
        return
    
    entry_price = get_close(current_bar)
    
    stop_loss_price = entry_price * (1.0 - LONG_STOP_LOSS_PCT)
    dynamic_tp = calculate_dynamic_take_profit(confidence)
    take_profit_price = entry_price * (1.0 + dynamic_tp)
    
    available_capital = 10000.0
    position_size = calculate_position_size(confidence, available_capital)
    quantity = position_size / entry_price
    
    print("OPEN LONG")
    print(entry_price)
    print(quantity)
    
    position_active = True
    position_entry_price = entry_price
    position_stop_loss = stop_loss_price
    position_take_profit = take_profit_price
    position_quantity = quantity
