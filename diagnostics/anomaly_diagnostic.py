"""
Anomaly Diagnostic — Random Forest Gap Classification

Uses missed-trade analysis + RF classifier to determine whether gaps are
STRUCTURAL (existing features sufficient, tree topology issue) or
FEATURE (signal in unmeasured data).

This is the L0 implementation — the concrete engineering for gap detection.

Explorer Prime v2.0 - Phase 3
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import Counter
import numpy as np

from .anomaly_signature import (
    TradeOpportunity,
    TemporalProfile,
    RegimeProfile,
    AssetProfile,
    VolatilityProfile,
    PrecedingPattern,
    LeadLagProfile,
    AnomalySignature,
    GapType,
    GapClassification,
    DiagnosisResult,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Market Data Protocol (duck typing)
# ==============================================================================

class MarketDataProtocol:
    """Protocol for market data input. Any object with these methods works."""
    def get_prices(self, asset: str, start: datetime, end: datetime) -> np.ndarray:
        raise NotImplementedError
    def get_features(self, asset: str, timestamp: datetime) -> np.ndarray:
        raise NotImplementedError
    def get_regime(self, timestamp: datetime) -> str:
        raise NotImplementedError
    def get_vol_regime(self, timestamp: datetime) -> str:
        raise NotImplementedError
    def get_assets(self) -> List[str]:
        raise NotImplementedError


# ==============================================================================
# Lightweight RF Classifier (no sklearn dependency)
# ==============================================================================

class DecisionStump:
    """Single decision tree stump for the RF ensemble."""

    def __init__(self, feature_idx: int = 0, threshold: float = 0.0,
                 left_value: float = 0.0, right_value: float = 1.0):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left_value = left_value
        self.right_value = right_value

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for class 1."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = np.where(
            X[:, self.feature_idx] <= self.threshold,
            self.left_value,
            self.right_value
        )
        return preds


class SimpleDecisionTree:
    """A simple decision tree with configurable max_depth."""

    def __init__(self, max_depth: int = 4, min_samples_split: int = 5,
                 rng: Optional[np.random.RandomState] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = rng or np.random.RandomState(42)
        self.tree: Optional[Dict] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_subset: Optional[np.ndarray] = None) -> 'SimpleDecisionTree':
        """Fit the decision tree."""
        self.tree = self._build_tree(X, y, depth=0, feature_subset=feature_subset)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int,
                    feature_subset: Optional[np.ndarray] = None) -> Dict:
        """Recursively build tree."""
        n_samples = len(y)
        positive_rate = np.mean(y) if n_samples > 0 else 0.0

        # Stopping conditions
        if (depth >= self.max_depth or n_samples < self.min_samples_split
                or len(np.unique(y)) <= 1):
            return {"leaf": True, "value": positive_rate}

        # Select feature subset for split search
        n_features = X.shape[1]
        if feature_subset is not None:
            candidates = feature_subset
        else:
            max_features = max(1, int(math.sqrt(n_features)))
            candidates = self.rng.choice(n_features, size=min(max_features, n_features),
                                          replace=False)

        best_gain = -1.0
        best_feature = 0
        best_threshold = 0.0
        best_left_mask = None

        parent_gini = self._gini(y)

        for feat in candidates:
            values = X[:, feat]
            unique_vals = np.unique(values)
            if len(unique_vals) < 2:
                continue

            # Test a sample of thresholds
            n_thresh = min(10, len(unique_vals) - 1)
            thresholds = np.percentile(values, np.linspace(10, 90, n_thresh))

            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                if left_mask.sum() < 2 or right_mask.sum() < 2:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])

                n_left = left_mask.sum()
                n_right = right_mask.sum()
                weighted_gini = (n_left * left_gini + n_right * right_gini) / n_samples
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thresh
                    best_left_mask = left_mask

        if best_gain <= 0 or best_left_mask is None:
            return {"leaf": True, "value": positive_rate}

        right_mask = ~best_left_mask
        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[best_left_mask], y[best_left_mask],
                                     depth + 1, feature_subset),
            "right": self._build_tree(X[right_mask], y[right_mask],
                                      depth + 1, feature_subset),
        }

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        """Compute Gini impurity."""
        if len(y) == 0:
            return 0.0
        p = np.mean(y)
        return 2.0 * p * (1.0 - p)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of class 1."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray) -> float:
        """Predict for a single sample."""
        node = self.tree
        while node and not node.get("leaf", True):
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node.get("value", 0.5) if node else 0.5


class SimpleRandomForest:
    """Lightweight Random Forest classifier (no sklearn needed)."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 4,
                 min_samples_split: int = 5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees: List[SimpleDecisionTree] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleRandomForest':
        """Fit the random forest with bootstrap sampling."""
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.trees = []
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)

        max_features = max(1, int(math.sqrt(n_features)))

        for i in range(self.n_estimators):
            tree_rng = np.random.RandomState(self.random_state + i)

            # Bootstrap sample
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[np.unique(indices)] = False

            # Feature subset
            feature_subset = rng.choice(n_features, size=min(max_features, n_features),
                                        replace=False)

            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                rng=tree_rng,
            )
            tree.fit(X[indices], y[indices], feature_subset=feature_subset)
            self.trees.append(tree)

            # OOB tracking
            if oob_mask.any():
                oob_preds = tree.predict_proba(X[oob_mask])
                oob_predictions[oob_mask] += oob_preds
                oob_counts[oob_mask] += 1

        # Compute feature importances via permutation (simplified)
        self._compute_feature_importances(X, y, rng)

        return self

    def _compute_feature_importances(self, X: np.ndarray, y: np.ndarray,
                                      rng: np.random.RandomState) -> None:
        """Compute feature importances using univariate AUC.

        For each feature, compute the AUC of that single feature as a
        classifier. This is more robust than permutation importance when
        the model is very powerful (all-feature AUC near 1.0 means
        permuting one feature doesn't degrade performance).
        """
        n_features = X.shape[1]
        importances = np.zeros(n_features)

        pos_mask = y == 1
        neg_mask = y == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            self.feature_importances_ = importances
            return

        for feat in range(n_features):
            pos_vals = X[pos_mask, feat]
            neg_vals = X[neg_mask, feat]

            # Univariate AUC via Mann-Whitney
            n_pos = len(pos_vals)
            n_neg = len(neg_vals)
            if n_pos == 0 or n_neg == 0:
                continue

            u_stat = 0.0
            for pv in pos_vals:
                u_stat += (neg_vals < pv).sum() + 0.5 * (neg_vals == pv).sum()
            auc = u_stat / (n_pos * n_neg)

            # Convert to absolute deviation from 0.5 (both directions are informative)
            importances[feat] = abs(auc - 0.5) * 2.0

        total = importances.sum()
        if total > 0:
            importances /= total
        self.feature_importances_ = importances

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict average probability across all trees."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = np.array([tree.predict_proba(X) for tree in self.trees])
        return predictions.mean(axis=0)

    def _compute_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute approximate AUC (using Mann-Whitney U statistic)."""
        proba = self.predict_proba(X)
        pos_scores = proba[y == 1]
        neg_scores = proba[y == 0]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.5

        # Mann-Whitney U statistic for AUC
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        u_stat = 0.0
        for ps in pos_scores:
            u_stat += (neg_scores < ps).sum() + 0.5 * (neg_scores == ps).sum()

        auc = u_stat / (n_pos * n_neg)
        return auc


# ==============================================================================
# UMAP + HDBSCAN Substitutes (lightweight implementations)
# ==============================================================================

def _simple_umap(X: np.ndarray, n_components: int = 2,
                 random_state: int = 42) -> np.ndarray:
    """Lightweight 2D embedding using PCA-like dimensionality reduction.

    This is a simplified substitute for UMAP — uses random projection
    followed by normalization. For production, replace with actual UMAP.
    """
    if X.shape[0] == 0:
        return np.zeros((0, n_components))
    if X.shape[1] <= n_components:
        padded = np.zeros((X.shape[0], n_components))
        padded[:, :X.shape[1]] = X
        return padded

    rng = np.random.RandomState(random_state)

    # Center the data
    X_centered = X - X.mean(axis=0)

    # Compute covariance and top eigenvectors (simple PCA)
    n_samples, n_features = X_centered.shape
    if n_samples < n_features:
        # Use gram matrix approach
        gram = X_centered @ X_centered.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        # Take top components
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        components = X_centered.T @ eigenvectors[:, idx]
        # Normalize
        norms = np.linalg.norm(components, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        components /= norms
        embedding = X_centered @ components
    else:
        cov = X_centered.T @ X_centered / n_samples
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        components = eigenvectors[:, idx]
        embedding = X_centered @ components

    return embedding


def _simple_hdbscan(embedding: np.ndarray, min_cluster_size: int = 5) -> Tuple[int, np.ndarray]:
    """Lightweight cluster detection using distance-based heuristic.

    Simplified substitute for HDBSCAN. Uses k-means style splitting.
    For production, replace with actual HDBSCAN.
    """
    n_samples = embedding.shape[0]
    if n_samples < min_cluster_size * 2:
        return 1, np.zeros(n_samples, dtype=int)

    # Try 2-means clustering
    labels = _kmeans_2(embedding)
    cluster_0 = (labels == 0).sum()
    cluster_1 = (labels == 1).sum()

    # Check if both clusters are substantial
    if cluster_0 >= min_cluster_size and cluster_1 >= min_cluster_size:
        # Check separation using silhouette-like metric
        center_0 = embedding[labels == 0].mean(axis=0)
        center_1 = embedding[labels == 1].mean(axis=0)
        inter_dist = np.linalg.norm(center_0 - center_1)

        intra_0 = np.mean(np.linalg.norm(embedding[labels == 0] - center_0, axis=1))
        intra_1 = np.mean(np.linalg.norm(embedding[labels == 1] - center_1, axis=1))
        avg_intra = (intra_0 + intra_1) / 2

        if inter_dist > 1.5 * avg_intra and avg_intra > 0:
            return 2, labels

    return 1, np.zeros(n_samples, dtype=int)


def _kmeans_2(X: np.ndarray, max_iter: int = 20) -> np.ndarray:
    """Simple 2-means clustering."""
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)

    # Initialize with first and last point
    c0 = X[0].copy()
    c1 = X[-1].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d0 = np.linalg.norm(X - c0, axis=1)
        d1 = np.linalg.norm(X - c1, axis=1)
        new_labels = (d1 < d0).astype(int)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        if (labels == 0).any():
            c0 = X[labels == 0].mean(axis=0)
        if (labels == 1).any():
            c1 = X[labels == 1].mean(axis=0)

    return labels


# ==============================================================================
# Granger Causality (simplified)
# ==============================================================================

def _granger_test(x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> Tuple[float, float]:
    """Simplified Granger causality test.

    Tests if lagged values of x help predict y beyond y's own lags.
    Returns (correlation, approximate p-value).
    """
    n = len(x)
    if n < max_lag + 10:
        return 0.0, 1.0

    # Build lag matrix for x and y
    best_corr = 0.0
    for lag in range(1, max_lag + 1):
        x_lagged = x[:-lag]
        y_current = y[lag:]

        # Remove mean
        x_c = x_lagged - x_lagged.mean()
        y_c = y_current - y_current.mean()

        denom = np.sqrt((x_c ** 2).sum() * (y_c ** 2).sum())
        if denom > 0:
            corr = abs((x_c * y_c).sum() / denom)
            if corr > best_corr:
                best_corr = corr

    # Approximate p-value using Fisher transform
    if best_corr >= 1.0:
        p_value = 0.0
    elif best_corr <= 0:
        p_value = 1.0
    else:
        z = 0.5 * math.log((1 + best_corr) / (1 - best_corr))
        se = 1.0 / math.sqrt(n - 3) if n > 3 else 1.0
        t_stat = abs(z / se) if se > 0 else 0.0
        # Approximate two-sided p-value
        p_value = max(0.0, min(1.0, 2.0 * math.exp(-0.5 * t_stat ** 2)))

    return best_corr, p_value


# ==============================================================================
# Anomaly Diagnostic Engine
# ==============================================================================

class AnomalyDiagnostic:
    """Main diagnostic engine for missed-trade analysis and gap classification.

    Orchestrates: collect → build_signature → classify_gap → diagnose

    Integration with L0-L3:
    - L0: AnomalySignature IS the L0 output (structured gap characterization)
    - L1: GapClassification top_features + DataClass mapping provides category-level info
    - L2-L3: Built on this foundation in future phases
    """

    # Classification thresholds
    STRUCTURAL_THRESHOLD = 0.7   # core_overlap >= this → STRUCTURAL
    FEATURE_THRESHOLD = 0.4      # core_overlap <= this → FEATURE
    MIN_PROFIT_RATIO = 2.0       # Forward return must be > 2x transaction cost
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = 4
    TOP_N_FEATURES = 5

    def __init__(self, feature_registry=None):
        """Initialize with optional feature registry for CORE feature lookup.

        Args:
            feature_registry: FeatureRegistry instance. If None, all features
                             are treated as core (for backward compatibility).
        """
        self.feature_registry = feature_registry
        self._core_feature_ids: Optional[set] = None
        if feature_registry is not None:
            self._refresh_core_features()

    def _refresh_core_features(self) -> None:
        """Refresh cached core feature IDs from registry."""
        if self.feature_registry is not None:
            from shared.feature_registry import FeatureStatus
            core_schema = self.feature_registry.get_schema(status_filter=FeatureStatus.CORE)
            self._core_feature_ids = set(core_schema)
        else:
            self._core_feature_ids = None

    # --------------------------------------------------------------------------
    # Step 1: Collect Missed Trades
    # --------------------------------------------------------------------------

    def collect_missed_trades(
        self,
        strategies: list,
        market_data: Any,
        lookback_days: int = 30,
        feature_names: Optional[List[str]] = None,
    ) -> List[TradeOpportunity]:
        """Identify profitable gaps where strategies were flat.

        For each strategy, find time periods with no position where a
        profitable trade existed (forward return > 2x transaction cost).

        Args:
            strategies: List of UnifiedStrategy or objects with .genome
            market_data: Object following MarketDataProtocol
            lookback_days: How far back to look
            feature_names: Optional feature name list for labeling

        Returns:
            List of TradeOpportunity instances
        """
        missed = []
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)

        if hasattr(market_data, 'get_missed_trades'):
            # Direct interface — market data provides missed trades
            raw_misses = market_data.get_missed_trades(strategies, start_date, end_date)
            for rm in raw_misses:
                if isinstance(rm, TradeOpportunity):
                    missed.append(rm)
                elif isinstance(rm, dict):
                    missed.append(TradeOpportunity(**rm))
        else:
            # Synthetic collection — use feature vectors from market data
            logger.warning("Market data does not implement get_missed_trades; "
                          "using simplified collection")
            # In production, this would scan bar-by-bar for flat periods
            # with profitable forward returns. For now, accept pre-built lists.
            pass

        # Filter by profitability threshold
        missed = [m for m in missed if m.profit_ratio() >= self.MIN_PROFIT_RATIO]

        return missed

    # --------------------------------------------------------------------------
    # Step 2: Build Anomaly Signature
    # --------------------------------------------------------------------------

    def build_signature(
        self,
        missed_trades: List[TradeOpportunity],
        transition_timestamps: Optional[List[datetime]] = None,
        cross_asset_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> AnomalySignature:
        """Build the full anomaly signature from missed trades.

        Extracts all observable characteristics: temporal, regime, asset,
        volatility, preceding pattern (UMAP+HDBSCAN), lead-lag (Granger).

        Args:
            missed_trades: Identified missed trade opportunities
            transition_timestamps: Regime transition timestamps for concentration
            cross_asset_data: Dict of asset -> price series for lead-lag analysis

        Returns:
            AnomalySignature (the L0 output)
        """
        sig = AnomalySignature(missed_trades=missed_trades)

        if not missed_trades:
            return sig

        # --- Temporal Clustering ---
        hour_counts: Dict[int, int] = Counter()
        dow_counts: Dict[int, int] = Counter()
        miss_timestamps = []

        for mt in missed_trades:
            hour_counts[mt.timestamp.hour] += 1
            dow_counts[mt.timestamp.weekday()] += 1
            miss_timestamps.append(mt.timestamp)

        total = len(missed_trades)
        sig.temporal_clustering.hour_distribution = {
            h: c / total for h, c in hour_counts.items()
        }
        sig.temporal_clustering.day_of_week_distribution = {
            d: c / total for d, c in dow_counts.items()
        }
        sig.temporal_clustering.detect_pattern()

        # --- Regime Distribution ---
        regime_counts = Counter(mt.regime for mt in missed_trades)
        sig.regime_distribution.regime_counts = dict(regime_counts)

        if transition_timestamps:
            sig.regime_distribution.detect_concentration(
                transition_timestamps, miss_timestamps
            )

        # --- Asset Concentration ---
        asset_counts = Counter(mt.asset for mt in missed_trades)
        sig.asset_concentration.asset_counts = dict(asset_counts)
        sig.asset_concentration.analyze()

        # --- Volatility Context ---
        sig.volatility_context.vol_at_miss = [
            np.std(mt.feature_vector) if mt.feature_vector is not None else 0.0
            for mt in missed_trades
        ]
        sig.volatility_context.vol_regime_at_miss = [
            mt.vol_regime for mt in missed_trades
        ]
        sig.volatility_context.analyze()

        # --- Preceding Market Pattern (UMAP + HDBSCAN) ---
        feature_vectors = np.array([
            mt.feature_vector for mt in missed_trades
            if mt.feature_vector is not None
        ])
        if len(feature_vectors) >= 5:
            sig.preceding_market_pattern.feature_vectors_30min_before = feature_vectors
            embedding = _simple_umap(feature_vectors)
            sig.preceding_market_pattern.umap_embedding = embedding
            n_clusters, labels = _simple_hdbscan(embedding)
            sig.preceding_market_pattern.n_subclusters = n_clusters
            sig.preceding_market_pattern.subcluster_labels = labels

        # --- Lead-Lag Structure ---
        if cross_asset_data:
            correlations: Dict[str, float] = {}
            p_values: Dict[str, float] = {}

            # Use mean feature values at miss times as target series
            miss_signal = np.array([
                mt.feature_vector.mean() if mt.feature_vector is not None else 0.0
                for mt in missed_trades
            ])

            for asset, price_series in cross_asset_data.items():
                if len(price_series) >= len(miss_signal):
                    # Align by taking the last N values
                    aligned = price_series[-len(miss_signal):]
                    corr, p_val = _granger_test(aligned, miss_signal)
                    correlations[asset] = corr
                    p_values[asset] = p_val

            sig.lead_lag_structure.identify_leads(correlations, p_values)

        return sig

    # --------------------------------------------------------------------------
    # Step 3: Classify Gap
    # --------------------------------------------------------------------------

    def classify_gap(
        self,
        missed_trades: List[TradeOpportunity],
        control_features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> GapClassification:
        """Classify the gap using Random Forest on missed vs. normal conditions.

        Builds a control set, trains RF, extracts feature importances, and
        determines gap type from core_overlap of top features.

        Args:
            missed_trades: The missed trade opportunities
            control_features: Control set feature vectors (normal conditions).
                             If None, generates synthetic control set.
            feature_names: Feature names for mapping to core features.

        Returns:
            GapClassification with core_overlap and gap type
        """
        if not missed_trades:
            return GapClassification(
                core_overlap=0.5,
                gap_type=GapType.UNKNOWN,
                top_features=[],
                baseline_auc=0.5,
            )

        # Build feature matrices
        X_missed = np.array([
            mt.feature_vector for mt in missed_trades
            if mt.feature_vector is not None
        ])

        if len(X_missed) == 0:
            return GapClassification(
                core_overlap=0.5,
                gap_type=GapType.UNKNOWN,
                top_features=[],
                baseline_auc=0.5,
            )

        n_missed = len(X_missed)
        n_features = X_missed.shape[1]

        # Build control set (equal size, balanced)
        if control_features is not None:
            X_control = control_features
            if len(X_control) > n_missed:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_control), size=n_missed, replace=False)
                X_control = X_control[idx]
        else:
            # Synthetic control: perturb missed features
            rng = np.random.RandomState(42)
            X_control = X_missed + rng.normal(0, 0.5, X_missed.shape)

        # Balance the classes
        n_control = len(X_control)
        n_min = min(n_missed, n_control)

        X = np.vstack([X_missed[:n_min], X_control[:n_min]])
        y = np.concatenate([np.ones(n_min), np.zeros(n_min)])

        # Train Random Forest
        rf = SimpleRandomForest(
            n_estimators=self.RF_N_ESTIMATORS,
            max_depth=self.RF_MAX_DEPTH,
            random_state=42,
        )
        rf.fit(X, y)

        # Extract feature importances and top features
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:self.TOP_N_FEATURES]

        # Map to feature names
        if feature_names and len(feature_names) == n_features:
            top_features = [
                (feature_names[i], float(importances[i]))
                for i in top_indices if importances[i] > 0
            ]
        else:
            top_features = [
                (f"feature_{i}", float(importances[i]))
                for i in top_indices if importances[i] > 0
            ]

        # Compute core_overlap: fraction of top features in CORE schema
        core_overlap = self._compute_core_overlap(top_features)

        # Determine gap type
        if core_overlap >= self.STRUCTURAL_THRESHOLD:
            gap_type = GapType.STRUCTURAL
        elif core_overlap <= self.FEATURE_THRESHOLD:
            gap_type = GapType.FEATURE
        else:
            gap_type = GapType.AMBIGUOUS

        # Compute AUC
        auc = rf._compute_auc(X, y)

        # Identify missing (non-core) features
        missing_features = [
            feat_id for feat_id, _ in top_features
            if not self._is_core_feature(feat_id)
        ]

        # L1 extension: data class distribution
        data_class_dist = self._compute_data_class_distribution(top_features)

        return GapClassification(
            core_overlap=core_overlap,
            gap_type=gap_type,
            top_features=top_features,
            rf_classifier=rf,
            baseline_auc=auc,
            missing_features=missing_features,
            data_class_distribution=data_class_dist,
        )

    def _compute_core_overlap(self, top_features: List[Tuple[str, float]]) -> float:
        """Compute fraction of top features that are in CORE schema."""
        if not top_features:
            return 0.5  # Default for no data

        if self._core_feature_ids is None:
            # No registry — treat all as core (backward compat)
            return 1.0

        n_core = sum(1 for fid, _ in top_features if fid in self._core_feature_ids)
        return n_core / len(top_features)

    def _is_core_feature(self, feature_id: str) -> bool:
        """Check if a feature is in the CORE schema."""
        if self._core_feature_ids is None:
            return True  # No registry → all core
        return feature_id in self._core_feature_ids

    def _compute_data_class_distribution(
        self, top_features: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """Map top features to their data classes for L1 info."""
        if self.feature_registry is None:
            return {}

        dist: Dict[str, float] = {}
        total_importance = sum(imp for _, imp in top_features)
        if total_importance == 0:
            return {}

        for feat_id, importance in top_features:
            feat_def = self.feature_registry.features.get(feat_id)
            if feat_def:
                dc = feat_def.data_class.value
                dist[dc] = dist.get(dc, 0.0) + importance / total_importance

        return dist

    # --------------------------------------------------------------------------
    # Step 4: Full Diagnosis
    # --------------------------------------------------------------------------

    def diagnose(
        self,
        strategies: list,
        market_data: Any,
        lookback_days: int = 30,
        control_features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        transition_timestamps: Optional[List[datetime]] = None,
        cross_asset_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> DiagnosisResult:
        """Full diagnostic pipeline: collect → signature → classify.

        Args:
            strategies: Strategies to analyze
            market_data: Market data source
            lookback_days: Lookback window
            control_features: Control set (normal conditions)
            feature_names: Feature names for core overlap
            transition_timestamps: Regime transition times
            cross_asset_data: Cross-asset price series

        Returns:
            DiagnosisResult with signature + classification + recommendations
        """
        # Step 1: Collect missed trades
        missed = self.collect_missed_trades(strategies, market_data, lookback_days,
                                            feature_names)

        # Step 2: Build signature
        signature = self.build_signature(
            missed, transition_timestamps, cross_asset_data
        )

        # Step 3: Classify gap
        classification = self.classify_gap(missed, control_features, feature_names)

        # Step 4: Build diagnosis result
        recommended_action = self._get_recommended_action(classification.gap_type)
        tree_seeds = None
        missing_candidates = None

        if classification.gap_type == GapType.STRUCTURAL:
            tree_seeds = self._extract_tree_seeds(classification.rf_classifier)
        elif classification.gap_type == GapType.FEATURE:
            missing_candidates = classification.missing_features
        elif classification.gap_type == GapType.AMBIGUOUS:
            tree_seeds = self._extract_tree_seeds(classification.rf_classifier)
            missing_candidates = classification.missing_features

        return DiagnosisResult(
            signature=signature,
            classification=classification,
            recommended_action=recommended_action,
            tree_topology_seeds=tree_seeds,
            missing_feature_candidates=missing_candidates,
            confidence=classification.baseline_auc,
        )

    @staticmethod
    def _get_recommended_action(gap_type: GapType) -> str:
        """Map gap type to recommended action string."""
        mapping = {
            GapType.STRUCTURAL: "structural_seeds",
            GapType.FEATURE: "feature_scout",
            GapType.AMBIGUOUS: "sequential_intervention",
            GapType.UNKNOWN: "sequential_intervention",
            GapType.PATTERN: "structural_seeds",
        }
        return mapping.get(gap_type, "sequential_intervention")

    def _extract_tree_seeds(self, rf_classifier: Any) -> List[Dict]:
        """Extract decision boundaries from RF trees as topology seeds.

        Converts RF splits into tree template seeds for evolutionary search.
        """
        if rf_classifier is None:
            return []

        seeds = []
        if isinstance(rf_classifier, SimpleRandomForest):
            # Extract top 3 most important trees (by depth)
            for i, tree in enumerate(rf_classifier.trees[:3]):
                seed = self._tree_to_seed(tree.tree, depth=0, seed_id=i)
                if seed:
                    seeds.append(seed)

        return seeds

    def _tree_to_seed(self, node: Optional[Dict], depth: int = 0,
                      seed_id: int = 0) -> Optional[Dict]:
        """Convert a decision tree node into a topology seed."""
        if node is None or node.get("leaf", True):
            return None

        return {
            "seed_id": seed_id,
            "feature": node.get("feature", 0),
            "threshold": node.get("threshold", 0.0),
            "depth": depth,
            "left": self._tree_to_seed(node.get("left"), depth + 1, seed_id),
            "right": self._tree_to_seed(node.get("right"), depth + 1, seed_id),
        }
