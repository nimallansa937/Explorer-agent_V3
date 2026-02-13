"""
Unified Feature Schema for EXPLORER PRIME

Defines the canonical 60-dimension feature vector used across:
- HIFA v2.0 meta-labeling
- Hinance shadow trading feature extraction
- Strategy generation conditioning

Feature Groups (60 total):
- Price Features (15): Returns, trends, support/resistance
- Volume Features (10): Volume patterns, VWAP, OBV
- Volatility Features (10): ATR, Bollinger, historical vol
- Momentum Features (10): RSI, MACD, momentum indicators
- Microstructure Features (8): Spread, depth, imbalance
- Regime Features (4): VIX-analog, trend state
- Time Features (3): Hour, day of week, month

This ensures consistent feature representation across all pipeline stages.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from enum import Enum


# ==============================================================================
# Feature Schema Definition
# ==============================================================================

class FeatureGroup(Enum):
    """Feature group categories."""
    PRICE = "price"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    MICROSTRUCTURE = "microstructure"
    REGIME = "regime"
    TIME = "time"


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""
    name: str
    group: FeatureGroup
    index: int
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_lookback: int = 20
    normalization: str = "zscore"     # "zscore", "minmax", "none"


# Complete feature schema (60 features)
FEATURE_SCHEMA: Dict[str, FeatureDefinition] = {}
FEATURE_DIMENSIONS = 60

# ==============================================================================
# Price Features (0-14, 15 features)
# ==============================================================================

_price_features = [
    FeatureDefinition("return_1bar", FeatureGroup.PRICE, 0, "1-bar return", -0.2, 0.2),
    FeatureDefinition("return_5bar", FeatureGroup.PRICE, 1, "5-bar return", -0.5, 0.5),
    FeatureDefinition("return_20bar", FeatureGroup.PRICE, 2, "20-bar return", -1.0, 1.0),
    FeatureDefinition("return_60bar", FeatureGroup.PRICE, 3, "60-bar return", -2.0, 2.0),
    FeatureDefinition("ema_cross_5_20", FeatureGroup.PRICE, 4, "EMA 5/20 crossover signal", -1, 1),
    FeatureDefinition("ema_cross_20_50", FeatureGroup.PRICE, 5, "EMA 20/50 crossover signal", -1, 1),
    FeatureDefinition("sma_distance_20", FeatureGroup.PRICE, 6, "Distance from 20 SMA (normalized)", -3, 3),
    FeatureDefinition("sma_distance_50", FeatureGroup.PRICE, 7, "Distance from 50 SMA (normalized)", -3, 3),
    FeatureDefinition("sma_distance_200", FeatureGroup.PRICE, 8, "Distance from 200 SMA (normalized)", -3, 3),
    FeatureDefinition("trend_strength", FeatureGroup.PRICE, 9, "ADX trend strength", 0, 100),
    FeatureDefinition("trend_direction", FeatureGroup.PRICE, 10, "Trend direction (+1/-1)", -1, 1),
    FeatureDefinition("support_distance", FeatureGroup.PRICE, 11, "Distance to nearest support", 0, 1),
    FeatureDefinition("resistance_distance", FeatureGroup.PRICE, 12, "Distance to nearest resistance", 0, 1),
    FeatureDefinition("pivot_position", FeatureGroup.PRICE, 13, "Position relative to pivot", -1, 1),
    FeatureDefinition("high_low_range", FeatureGroup.PRICE, 14, "Normalized high-low range", 0, 1),
]

# ==============================================================================
# Volume Features (15-24, 10 features)
# ==============================================================================

_volume_features = [
    FeatureDefinition("volume_ratio_20", FeatureGroup.VOLUME, 15, "Volume / 20-period avg", 0, 10),
    FeatureDefinition("volume_trend", FeatureGroup.VOLUME, 16, "Volume trend (slope)", -1, 1),
    FeatureDefinition("obv_change", FeatureGroup.VOLUME, 17, "OBV change normalized", -1, 1),
    FeatureDefinition("vwap_distance", FeatureGroup.VOLUME, 18, "Distance from VWAP", -3, 3),
    FeatureDefinition("volume_profile_poc", FeatureGroup.VOLUME, 19, "Distance from Volume POC", -1, 1),
    FeatureDefinition("buy_volume_ratio", FeatureGroup.VOLUME, 20, "Buy volume / total volume", 0, 1),
    FeatureDefinition("volume_momentum", FeatureGroup.VOLUME, 21, "Volume momentum indicator", -1, 1),
    FeatureDefinition("accumulation_dist", FeatureGroup.VOLUME, 22, "Accumulation/Distribution line", -1, 1),
    FeatureDefinition("chaikin_mf", FeatureGroup.VOLUME, 23, "Chaikin Money Flow", -1, 1),
    FeatureDefinition("mfi", FeatureGroup.VOLUME, 24, "Money Flow Index", 0, 100),
]

# ==============================================================================
# Volatility Features (25-34, 10 features)
# ==============================================================================

_volatility_features = [
    FeatureDefinition("atr_14", FeatureGroup.VOLATILITY, 25, "ATR 14 normalized", 0, 1),
    FeatureDefinition("atr_ratio", FeatureGroup.VOLATILITY, 26, "ATR / 20-period avg ATR", 0, 5),
    FeatureDefinition("bb_width", FeatureGroup.VOLATILITY, 27, "Bollinger Band width", 0, 1),
    FeatureDefinition("bb_position", FeatureGroup.VOLATILITY, 28, "Position within BB (-1 to 1)", -1.5, 1.5),
    FeatureDefinition("keltner_position", FeatureGroup.VOLATILITY, 29, "Position within Keltner Channel", -1.5, 1.5),
    FeatureDefinition("historical_vol_20", FeatureGroup.VOLATILITY, 30, "20-period historical volatility", 0, 2),
    FeatureDefinition("vol_of_vol", FeatureGroup.VOLATILITY, 31, "Volatility of volatility", 0, 2),
    FeatureDefinition("vol_regime", FeatureGroup.VOLATILITY, 32, "Volatility regime (low/med/high)", 0, 2),
    FeatureDefinition("parkinson_vol", FeatureGroup.VOLATILITY, 33, "Parkinson volatility estimator", 0, 2),
    FeatureDefinition("garman_klass", FeatureGroup.VOLATILITY, 34, "Garman-Klass volatility", 0, 2),
]

# ==============================================================================
# Momentum Features (35-44, 10 features)
# ==============================================================================

_momentum_features = [
    FeatureDefinition("rsi_14", FeatureGroup.MOMENTUM, 35, "RSI 14", 0, 100),
    FeatureDefinition("rsi_divergence", FeatureGroup.MOMENTUM, 36, "RSI divergence signal", -1, 1),
    FeatureDefinition("macd_signal", FeatureGroup.MOMENTUM, 37, "MACD signal line cross", -1, 1),
    FeatureDefinition("macd_histogram", FeatureGroup.MOMENTUM, 38, "MACD histogram normalized", -1, 1),
    FeatureDefinition("stochastic_k", FeatureGroup.MOMENTUM, 39, "Stochastic %K", 0, 100),
    FeatureDefinition("stochastic_d", FeatureGroup.MOMENTUM, 40, "Stochastic %D", 0, 100),
    FeatureDefinition("cci", FeatureGroup.MOMENTUM, 41, "CCI normalized", -2, 2),
    FeatureDefinition("williams_r", FeatureGroup.MOMENTUM, 42, "Williams %R", -100, 0),
    FeatureDefinition("roc_10", FeatureGroup.MOMENTUM, 43, "Rate of Change 10", -0.5, 0.5),
    FeatureDefinition("ultimate_osc", FeatureGroup.MOMENTUM, 44, "Ultimate Oscillator", 0, 100),
]

# ==============================================================================
# Microstructure Features (45-52, 8 features)
# ==============================================================================

_microstructure_features = [
    FeatureDefinition("bid_ask_spread", FeatureGroup.MICROSTRUCTURE, 45, "Bid-ask spread (bps)", 0, 100),
    FeatureDefinition("order_imbalance", FeatureGroup.MICROSTRUCTURE, 46, "Order book imbalance", -1, 1),
    FeatureDefinition("depth_ratio", FeatureGroup.MICROSTRUCTURE, 47, "Bid depth / Ask depth", 0, 5),
    FeatureDefinition("trade_intensity", FeatureGroup.MICROSTRUCTURE, 48, "Trade arrival rate", 0, 10),
    FeatureDefinition("quote_intensity", FeatureGroup.MICROSTRUCTURE, 49, "Quote update rate", 0, 10),
    FeatureDefinition("effective_spread", FeatureGroup.MICROSTRUCTURE, 50, "Effective spread", 0, 100),
    FeatureDefinition("price_impact", FeatureGroup.MICROSTRUCTURE, 51, "Estimated price impact", 0, 1),
    FeatureDefinition("kyle_lambda", FeatureGroup.MICROSTRUCTURE, 52, "Kyle's lambda (market depth)", 0, 1),
]

# ==============================================================================
# Regime Features (53-56, 4 features)
# ==============================================================================

_regime_features = [
    FeatureDefinition("vix_analog", FeatureGroup.REGIME, 53, "Crypto VIX analog", 0, 100),
    FeatureDefinition("regime_state", FeatureGroup.REGIME, 54, "Regime state (0=normal, 1=elevated, 2=crisis)", 0, 2),
    FeatureDefinition("correlation_btc", FeatureGroup.REGIME, 55, "Rolling correlation with BTC", -1, 1),
    FeatureDefinition("market_beta", FeatureGroup.REGIME, 56, "Rolling beta to market", -2, 2),
]

# ==============================================================================
# Time Features (57-59, 3 features)
# ==============================================================================

_time_features = [
    FeatureDefinition("hour_sin", FeatureGroup.TIME, 57, "Hour of day (sine encoded)", -1, 1),
    FeatureDefinition("hour_cos", FeatureGroup.TIME, 58, "Hour of day (cosine encoded)", -1, 1),
    FeatureDefinition("day_of_week", FeatureGroup.TIME, 59, "Day of week (0-6)", 0, 6),
]

# Build complete schema
for feature_list in [_price_features, _volume_features, _volatility_features,
                     _momentum_features, _microstructure_features, _regime_features,
                     _time_features]:
    for feature in feature_list:
        FEATURE_SCHEMA[feature.name] = feature


# ==============================================================================
# Feature Vector Class
# ==============================================================================

@dataclass
class FeatureVector:
    """
    60-dimensional feature vector with validation and utilities.

    Provides:
    - Type-safe feature storage
    - Validation against schema
    - Serialization/deserialization
    - Feature group slicing
    """
    values: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_DIMENSIONS))
    timestamp: Optional[Any] = None   # pd.Timestamp or datetime
    asset: str = "BTCUSDT"

    def __post_init__(self):
        """Validate feature vector dimensions."""
        if len(self.values) != FEATURE_DIMENSIONS:
            raise ValueError(f"Feature vector must have {FEATURE_DIMENSIONS} dimensions, got {len(self.values)}")
        self.values = np.array(self.values, dtype=np.float32)

    def __getitem__(self, key: str) -> float:
        """Get feature by name."""
        if key not in FEATURE_SCHEMA:
            raise KeyError(f"Unknown feature: {key}")
        return float(self.values[FEATURE_SCHEMA[key].index])

    def __setitem__(self, key: str, value: float):
        """Set feature by name."""
        if key not in FEATURE_SCHEMA:
            raise KeyError(f"Unknown feature: {key}")
        self.values[FEATURE_SCHEMA[key].index] = value

    def get_group(self, group: FeatureGroup) -> Dict[str, float]:
        """Get all features in a group."""
        return {
            name: float(self.values[feat.index])
            for name, feat in FEATURE_SCHEMA.items()
            if feat.group == group
        }

    def get_price_features(self) -> Dict[str, float]:
        return self.get_group(FeatureGroup.PRICE)

    def get_volume_features(self) -> Dict[str, float]:
        return self.get_group(FeatureGroup.VOLUME)

    def get_volatility_features(self) -> Dict[str, float]:
        return self.get_group(FeatureGroup.VOLATILITY)

    def get_momentum_features(self) -> Dict[str, float]:
        return self.get_group(FeatureGroup.MOMENTUM)

    def get_microstructure_features(self) -> Dict[str, float]:
        return self.get_group(FeatureGroup.MICROSTRUCTURE)

    def get_regime_features(self) -> Dict[str, float]:
        return self.get_group(FeatureGroup.REGIME)

    def get_time_features(self) -> Dict[str, float]:
        return self.get_group(FeatureGroup.TIME)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with feature names."""
        return {name: float(self.values[feat.index]) for name, feat in FEATURE_SCHEMA.items()}

    def to_array(self) -> np.ndarray:
        """Get raw numpy array."""
        return self.values.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, float], **kwargs) -> 'FeatureVector':
        """Create from dictionary."""
        values = np.zeros(FEATURE_DIMENSIONS)
        for name, value in data.items():
            if name in FEATURE_SCHEMA:
                values[FEATURE_SCHEMA[name].index] = value
        return cls(values=values, **kwargs)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate feature values against schema bounds."""
        errors = []
        for name, feat in FEATURE_SCHEMA.items():
            value = self.values[feat.index]

            if np.isnan(value):
                errors.append(f"{name}: NaN value")
            elif np.isinf(value):
                errors.append(f"{name}: Infinite value")
            elif feat.min_value is not None and value < feat.min_value:
                errors.append(f"{name}: {value} < min {feat.min_value}")
            elif feat.max_value is not None and value > feat.max_value:
                errors.append(f"{name}: {value} > max {feat.max_value}")

        return len(errors) == 0, errors

    def clip_to_bounds(self) -> 'FeatureVector':
        """Clip values to schema bounds."""
        clipped = self.values.copy()
        for name, feat in FEATURE_SCHEMA.items():
            idx = feat.index
            if feat.min_value is not None:
                clipped[idx] = max(clipped[idx], feat.min_value)
            if feat.max_value is not None:
                clipped[idx] = min(clipped[idx], feat.max_value)
        return FeatureVector(values=clipped, timestamp=self.timestamp, asset=self.asset)


# ==============================================================================
# Feature Extraction Functions
# ==============================================================================

def extract_features(
    ohlcv: Any,  # pd.DataFrame with OHLCV columns
    orderbook: Optional[Any] = None,
    regime_state: Optional[int] = None,
) -> FeatureVector:
    """
    Extract 60-dimensional feature vector from market data.

    Args:
        ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        orderbook: Optional orderbook data for microstructure features
        regime_state: Optional regime override (0=normal, 1=elevated, 2=crisis)

    Returns:
        FeatureVector with all 60 features computed
    """
    import pandas as pd

    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame")

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in ohlcv.columns:
            raise ValueError(f"Missing required column: {col}")

    if len(ohlcv) < 200:
        raise ValueError("Need at least 200 bars for feature extraction")

    values = np.zeros(FEATURE_DIMENSIONS, dtype=np.float32)
    close = ohlcv['close'].values
    high = ohlcv['high'].values
    low = ohlcv['low'].values
    volume = ohlcv['volume'].values

    # Price features (0-14)
    values[0] = (close[-1] / close[-2] - 1) if len(close) > 1 else 0  # return_1bar
    values[1] = (close[-1] / close[-5] - 1) if len(close) > 5 else 0  # return_5bar
    values[2] = (close[-1] / close[-20] - 1) if len(close) > 20 else 0  # return_20bar
    values[3] = (close[-1] / close[-60] - 1) if len(close) > 60 else 0  # return_60bar

    # EMAs
    ema5 = _ema(close, 5)
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    sma200 = np.mean(close[-200:]) if len(close) >= 200 else close[-1]

    values[4] = 1 if ema5[-1] > ema20[-1] else -1  # ema_cross_5_20
    values[5] = 1 if ema20[-1] > ema50[-1] else -1  # ema_cross_20_50
    values[6] = (close[-1] - ema20[-1]) / (np.std(close[-20:]) + 1e-8)  # sma_distance_20
    values[7] = (close[-1] - ema50[-1]) / (np.std(close[-50:]) + 1e-8)  # sma_distance_50
    values[8] = (close[-1] - sma200) / (np.std(close[-200:]) + 1e-8)  # sma_distance_200

    # Trend strength (simplified ADX)
    tr = _true_range(high, low, close)
    atr = np.mean(tr[-14:])
    values[9] = min(atr / (close[-1] + 1e-8) * 1000, 100)  # trend_strength
    values[10] = 1 if close[-1] > close[-20] else -1  # trend_direction

    # Support/resistance (simplified)
    recent_low = np.min(low[-20:])
    recent_high = np.max(high[-20:])
    price_range = recent_high - recent_low + 1e-8
    values[11] = (close[-1] - recent_low) / price_range  # support_distance
    values[12] = (recent_high - close[-1]) / price_range  # resistance_distance
    values[13] = 2 * (close[-1] - recent_low) / price_range - 1  # pivot_position
    values[14] = (high[-1] - low[-1]) / (close[-1] + 1e-8)  # high_low_range

    # Volume features (15-24)
    vol_avg = np.mean(volume[-20:]) + 1e-8
    values[15] = volume[-1] / vol_avg  # volume_ratio_20
    values[16] = (volume[-1] - volume[-5]) / (np.std(volume[-20:]) + 1e-8)  # volume_trend

    # OBV simplified
    obv = np.cumsum(np.sign(np.diff(close, prepend=close[0])) * volume)
    values[17] = (obv[-1] - obv[-20]) / (np.std(obv[-20:]) + 1e-8)  # obv_change

    # VWAP
    vwap = np.cumsum(close * volume) / (np.cumsum(volume) + 1e-8)
    values[18] = (close[-1] - vwap[-1]) / (np.std(close[-20:]) + 1e-8)  # vwap_distance

    values[19] = 0  # volume_profile_poc (placeholder)
    values[20] = 0.5  # buy_volume_ratio (placeholder)
    values[21] = (volume[-1] - volume[-5]) / (vol_avg + 1e-8)  # volume_momentum
    values[22] = 0  # accumulation_dist (placeholder)
    values[23] = 0  # chaikin_mf (placeholder)
    values[24] = 50  # mfi (placeholder)

    # Volatility features (25-34)
    atr_14 = np.mean(tr[-14:])
    atr_avg = np.mean(tr[-60:]) + 1e-8
    values[25] = atr_14 / (close[-1] + 1e-8)  # atr_14
    values[26] = atr_14 / atr_avg  # atr_ratio

    # Bollinger Bands
    bb_mid = np.mean(close[-20:])
    bb_std = np.std(close[-20:])
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    values[27] = bb_width / (close[-1] + 1e-8)  # bb_width
    values[28] = (close[-1] - bb_mid) / (bb_std + 1e-8)  # bb_position
    values[29] = values[28]  # keltner_position (approximation)

    hist_vol = np.std(np.diff(np.log(close[-21:]))) * np.sqrt(252)
    values[30] = hist_vol  # historical_vol_20
    values[31] = np.std([np.std(np.diff(np.log(close[i:i+20]))) for i in range(-60, -20, 5)])  # vol_of_vol
    values[32] = 0 if hist_vol < 0.3 else (1 if hist_vol < 0.6 else 2)  # vol_regime
    values[33] = hist_vol  # parkinson_vol (approximation)
    values[34] = hist_vol  # garman_klass (approximation)

    # Momentum features (35-44)
    rsi = _rsi(close, 14)
    values[35] = rsi  # rsi_14
    values[36] = 0  # rsi_divergence (placeholder)

    # MACD
    macd_line = _ema(close, 12) - _ema(close, 26)
    macd_signal = _ema(macd_line, 9)
    macd_hist = macd_line - macd_signal
    values[37] = 1 if macd_line[-1] > macd_signal[-1] else -1  # macd_signal
    values[38] = macd_hist[-1] / (np.std(macd_hist[-20:]) + 1e-8)  # macd_histogram

    # Stochastic
    lowest_low = np.min(low[-14:])
    highest_high = np.max(high[-14:])
    stoch_k = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low + 1e-8)
    values[39] = stoch_k  # stochastic_k
    values[40] = stoch_k  # stochastic_d (approximation)

    values[41] = (close[-1] - np.mean(close[-20:])) / (0.015 * np.std(close[-20:]) + 1e-8)  # cci
    values[42] = -100 * (highest_high - close[-1]) / (highest_high - lowest_low + 1e-8)  # williams_r
    values[43] = (close[-1] / close[-10] - 1)  # roc_10
    values[44] = 50  # ultimate_osc (placeholder)

    # Microstructure features (45-52) - placeholders without orderbook
    if orderbook is not None:
        # Extract from orderbook if available
        pass
    values[45] = 5  # bid_ask_spread
    values[46] = 0  # order_imbalance
    values[47] = 1  # depth_ratio
    values[48] = 1  # trade_intensity
    values[49] = 1  # quote_intensity
    values[50] = 5  # effective_spread
    values[51] = 0.01  # price_impact
    values[52] = 0.1  # kyle_lambda

    # Regime features (53-56)
    vix_analog = hist_vol * 100
    values[53] = min(vix_analog, 100)  # vix_analog
    values[54] = regime_state if regime_state is not None else (0 if vix_analog < 20 else (1 if vix_analog < 30 else 2))  # regime_state
    values[55] = 1  # correlation_btc (placeholder)
    values[56] = 1  # market_beta (placeholder)

    # Time features (57-59)
    if hasattr(ohlcv.index[-1], 'hour'):
        hour = ohlcv.index[-1].hour
        values[57] = np.sin(2 * np.pi * hour / 24)  # hour_sin
        values[58] = np.cos(2 * np.pi * hour / 24)  # hour_cos
        values[59] = ohlcv.index[-1].dayofweek  # day_of_week
    else:
        values[57] = 0
        values[58] = 1
        values[59] = 0

    return FeatureVector(
        values=values,
        timestamp=ohlcv.index[-1] if hasattr(ohlcv.index[-1], 'timestamp') else None,
    )


def validate_features(features: FeatureVector) -> Tuple[bool, List[str]]:
    """
    Validate feature vector against schema.

    Args:
        features: FeatureVector to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    return features.validate()


# ==============================================================================
# Helper Functions
# ==============================================================================

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Compute exponential moving average."""
    alpha = 2 / (period + 1)
    ema = np.zeros_like(data, dtype=np.float64)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Compute true range."""
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    return tr


def _rsi(close: np.ndarray, period: int = 14) -> float:
    """Compute RSI."""
    deltas = np.diff(close[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ==============================================================================
# Feature Group Utilities
# ==============================================================================

def get_feature_indices(group: FeatureGroup) -> List[int]:
    """Get indices for a feature group."""
    return [feat.index for feat in FEATURE_SCHEMA.values() if feat.group == group]


def get_feature_names(group: Optional[FeatureGroup] = None) -> List[str]:
    """Get feature names, optionally filtered by group."""
    if group is None:
        return list(FEATURE_SCHEMA.keys())
    return [name for name, feat in FEATURE_SCHEMA.items() if feat.group == group]


def get_feature_descriptions() -> Dict[str, str]:
    """Get all feature descriptions."""
    return {name: feat.description for name, feat in FEATURE_SCHEMA.items()}
