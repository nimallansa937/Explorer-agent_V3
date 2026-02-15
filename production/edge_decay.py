"""
Edge Decay Detector — Kalman Filter for Sharpe Drift Estimation

Models each strategy's true Sharpe ratio as a drifting latent variable.
Produces calibrated posterior probabilities of edge decay.

Key distinction: Differentiates genuine edge decay from regime suppression.
A strategy that is strong in other regimes but weak in the current one
is regime-suppressed, not decaying.

Explorer Prime v2.0 - Phase 7
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


class EdgeDecayDetector:
    """Kalman filter tracking a strategy's true Sharpe ratio.

    State model:
      mu_{t+1} = mu_t + w_t      where w_t ~ N(0, drift_var)
      y_t = mu_t / sqrt(252) + v_t  where v_t ~ N(0, obs_var)

    The drift_var controls how fast the filter adapts to changing edges.
    Calibrated from MetaLearningSignal: set so that Sharpe decline 1.0→0.0
    over median_ttf days produces decay_probability > 0.7 within 0.7 * median_ttf days.

    Regime-conditioned decay separates genuine decay from regime suppression.
    """

    DEFAULT_DRIFT_VAR: float = 0.001
    DEFAULT_OBS_VAR: float = 0.5
    REGIME_SUPPRESSION_STRONG_THRESHOLD: float = 0.5
    REGIME_SUPPRESSION_WEAK_THRESHOLD: float = 0.2
    REGIME_SUPPRESSION_PROBABILITY: float = 0.15

    def __init__(
        self,
        initial_sharpe: float = 1.0,
        initial_sigma2: float = 0.1,
        drift_var: float = 0.001,
        obs_var: float = 0.5,
    ):
        self.mu = initial_sharpe
        self.sigma2 = initial_sigma2
        self.drift_var = drift_var
        self.obs_var = obs_var

        # Per-regime Sharpe history
        self.regime_sharpe_history: Dict[str, List[float]] = {}

    def update(
        self,
        daily_return: float,
        current_regime: str,
        annualization: int = 252,
    ) -> Tuple[float, float]:
        """Kalman filter update step.

        Args:
            daily_return: Today's strategy return
            current_regime: Current market regime
            annualization: Trading days per year

        Returns:
            (estimated_sharpe, uncertainty)
        """
        # Prediction step: uncertainty grows by drift
        self.sigma2 += self.drift_var

        # Update step
        sqrt_annual = math.sqrt(annualization)
        expected_return = self.mu / sqrt_annual
        residual = daily_return - expected_return

        # Kalman gain
        kalman_gain = self.sigma2 / (self.sigma2 + self.obs_var)

        # State update
        self.mu += kalman_gain * residual * sqrt_annual
        self.sigma2 *= (1 - kalman_gain)

        # Record regime history
        if current_regime not in self.regime_sharpe_history:
            self.regime_sharpe_history[current_regime] = []
        self.regime_sharpe_history[current_regime].append(self.mu)

        return (self.mu, self.sigma2)

    def decay_probability(self, threshold_sharpe: float = 0.3) -> float:
        """Posterior probability that true Sharpe is below threshold.

        P(true_sharpe < threshold | observations)

        Args:
            threshold_sharpe: The Sharpe threshold below which we consider decay

        Returns:
            Probability in [0, 1]
        """
        if self.sigma2 <= 0:
            return 1.0 if self.mu < threshold_sharpe else 0.0

        z = (threshold_sharpe - self.mu) / math.sqrt(self.sigma2)
        return _norm_cdf(z)

    def regime_conditioned_decay(self, current_regime: str) -> float:
        """Regime-aware decay probability.

        If strategy is strong in OTHER regimes but weak in current regime,
        this is regime suppression → return low probability (0.15).
        Otherwise return standard decay_probability().

        Args:
            current_regime: Current market regime

        Returns:
            Decay probability conditioned on regime context
        """
        # Need history in multiple regimes for comparison
        other_regimes = [
            regime for regime in self.regime_sharpe_history
            if regime != current_regime
        ]

        if not other_regimes:
            return self.decay_probability()

        # Average Sharpe in other regimes
        other_sharpes = []
        for regime in other_regimes:
            history = self.regime_sharpe_history[regime]
            if history:
                other_sharpes.append(np.mean(history[-20:]))  # Recent 20 obs

        if not other_sharpes:
            return self.decay_probability()

        avg_other = np.mean(other_sharpes)

        # Average Sharpe in current regime
        current_history = self.regime_sharpe_history.get(current_regime, [])
        if not current_history:
            return self.decay_probability()

        avg_current = np.mean(current_history[-20:])

        # Regime suppression: strong elsewhere, weak here
        if (avg_other > self.REGIME_SUPPRESSION_STRONG_THRESHOLD and
                avg_current < self.REGIME_SUPPRESSION_WEAK_THRESHOLD):
            return self.REGIME_SUPPRESSION_PROBABILITY

        return self.decay_probability()

    def calibrate_drift_var(self, median_ttf: float) -> None:
        """Calibrate drift_var so Kalman filter matches empirical decay speed.

        Binary search: find drift_var where Sharpe decline 1.0→0.0 over
        median_ttf simulated days produces decay_prob > 0.7 at 0.7 * median_ttf days.

        Args:
            median_ttf: Median time-to-failure from MetaLearningSignal
        """
        target_day = int(0.7 * median_ttf)
        if target_day <= 0:
            return

        lo, hi = 1e-6, 0.1
        for _ in range(50):  # Binary search iterations
            mid = (lo + hi) / 2.0

            # Simulate decay with this drift_var
            prob = self._simulate_decay_detection(
                drift_var=mid,
                sharpe_decline_days=int(median_ttf),
                check_day=target_day,
            )

            if prob > 0.7:
                hi = mid  # drift_var too high, detector too sensitive
            else:
                lo = mid  # drift_var too low, detector too slow

        self.drift_var = (lo + hi) / 2.0

    def _simulate_decay_detection(
        self,
        drift_var: float,
        sharpe_decline_days: int,
        check_day: int,
    ) -> float:
        """Simulate Sharpe decline and check detection at given day.

        Creates a synthetic scenario: Sharpe declines linearly from 1.0 to 0.0
        over sharpe_decline_days, and returns the decay probability at check_day.
        """
        sim = EdgeDecayDetector(
            initial_sharpe=1.0,
            initial_sigma2=0.1,
            drift_var=drift_var,
            obs_var=self.obs_var,
        )

        rng = np.random.RandomState(42)

        for day in range(check_day):
            # True Sharpe declines linearly
            true_sharpe = 1.0 - day / max(1, sharpe_decline_days)

            # Simulate daily return with noise
            daily_return = true_sharpe / math.sqrt(252) + rng.randn() * math.sqrt(self.obs_var)
            sim.update(daily_return, "SIM")

        return sim.decay_probability()


# ==============================================================================
# Lightweight normal CDF (no scipy dependency)
# ==============================================================================

def _norm_cdf(z: float) -> float:
    """Standard normal CDF using error function approximation."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
