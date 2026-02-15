"""
Meta-Learning Signal — Pipeline Parameter Calibration

Analyzes time-to-failure distributions to calibrate pipeline parameters:
- Shadow trading minimum duration
- Edge decay detector drift_var
- Failure archive time decay half-life

Also detects bimodal failure distributions (overfit vs genuine edge decay).

Explorer Prime v2.0 - Phase 5
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

from .failure_archive import FailureArchive, FailureDistribution


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class BimodalAnalysis:
    """Result of bimodality detection in time-to-failure distribution."""
    is_bimodal: bool = False
    dip_statistic: float = 0.0
    p_value: float = 1.0
    short_mode_median: Optional[float] = None   # Days (overfit strategies)
    long_mode_median: Optional[float] = None    # Days (genuine edge decay)
    recommended_gate_adjustments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineCalibration:
    """Recommended pipeline parameter changes from meta-learning.

    These are RECOMMENDATIONS — applied only after human review.
    """
    shadow_min_duration: int = 14              # Days
    drift_var: float = 0.01                    # For edge decay detector
    archive_half_life: float = 120.0           # Days
    bimodal_gate_adjustments: Optional[Dict[str, Any]] = None
    characteristic_decay_timescale: float = 50.0
    confidence: float = 0.0                    # How reliable the estimates are


# ==============================================================================
# Meta-Learning Signal
# ==============================================================================

class MetaLearningSignal:
    """Computes meta-learning signals from failure archive.

    Three calibration channels:
    1. Shadow trading duration → min(14, median_ttf * 0.3)
    2. Edge decay drift_var → calibrated to detect Sharpe decline
    3. Archive half-life → ~2x median_ttf

    Also detects bimodal failure distributions.
    """

    MIN_RECORDS_FOR_ANALYSIS: int = 10
    DEFAULT_SHADOW_DURATION: int = 14
    DEFAULT_DRIFT_VAR: float = 0.01
    DEFAULT_ARCHIVE_HALF_LIFE: float = 120.0

    def __init__(self, failure_archive: FailureArchive):
        self.archive = failure_archive

    def compute_characteristic_decay_timescale(self) -> float:
        """Compute median time-to-failure as the market's characteristic
        non-stationarity timescale.

        If median TTF is 50 days, edges last about 50 days on average.
        """
        dist = self.archive.get_failure_distribution()
        if dist.n_records < self.MIN_RECORDS_FOR_ANALYSIS:
            return 50.0  # Default

        ttf_values = sorted(dist.ttf_values)
        if not ttf_values:
            return 50.0

        return float(ttf_values[len(ttf_values) // 2])

    def detect_bimodal_failure(self) -> Optional[BimodalAnalysis]:
        """Test time-to-failure distribution for bimodality.

        Uses a simplified Hartigan's dip test proxy:
        checks if the distribution has two distinct modes by looking
        for a valley between two peaks in the histogram.

        Returns:
            BimodalAnalysis if enough data, None otherwise
        """
        dist = self.archive.get_failure_distribution()
        if dist.n_records < self.MIN_RECORDS_FOR_ANALYSIS:
            return None

        ttf_values = np.array(dist.ttf_values, dtype=float)
        if len(ttf_values) < 10:
            return None

        # Simplified bimodality detection using histogram analysis
        is_bimodal, dip_stat, p_value = self._hartigan_dip_proxy(ttf_values)

        result = BimodalAnalysis(
            is_bimodal=is_bimodal,
            dip_statistic=dip_stat,
            p_value=p_value,
        )

        if is_bimodal:
            # Split into two modes at the valley point
            short_mode, long_mode = self._split_modes(ttf_values)
            if short_mode is not None and long_mode is not None:
                result.short_mode_median = float(np.median(short_mode))
                result.long_mode_median = float(np.median(long_mode))

                # Recommend gate adjustments
                result.recommended_gate_adjustments = {
                    "tighten_validation": True,
                    "short_mode_median": result.short_mode_median,
                    "long_mode_median": result.long_mode_median,
                    "advice": (
                        f"Short-lived failures (median ~{result.short_mode_median:.0f} days): "
                        f"likely overfit → tighten HIFA validation gates. "
                        f"Long-lived failures (median ~{result.long_mode_median:.0f} days): "
                        f"genuine edge decay → improve retirement detection."
                    ),
                }

        return result

    def get_pipeline_calibration(self) -> PipelineCalibration:
        """Aggregate all meta-learning signals into parameter recommendations.

        Returns:
            PipelineCalibration with recommended parameter changes
        """
        median_ttf = self.compute_characteristic_decay_timescale()

        # Calibrate shadow duration: min(14, median_ttf * 0.3)
        shadow_duration = min(self.DEFAULT_SHADOW_DURATION,
                              max(7, int(median_ttf * 0.3)))

        # Calibrate drift_var for edge decay detection
        # Goal: Sharpe decline from 1.0→0.0 over median_ttf days should
        # produce decay_probability > 0.7 within 0.7 * median_ttf days
        drift_var = self._calibrate_drift_var(median_ttf)

        # Archive half-life: ~2x median_ttf
        archive_half_life = max(60.0, median_ttf * 2.0)

        # Check for bimodality
        bimodal = self.detect_bimodal_failure()
        bimodal_adjustments = None
        if bimodal and bimodal.is_bimodal:
            bimodal_adjustments = bimodal.recommended_gate_adjustments

        # Confidence: higher with more records
        n = self.archive.get_failure_distribution().n_records
        confidence = min(1.0, n / 100.0)

        return PipelineCalibration(
            shadow_min_duration=shadow_duration,
            drift_var=drift_var,
            archive_half_life=archive_half_life,
            bimodal_gate_adjustments=bimodal_adjustments,
            characteristic_decay_timescale=median_ttf,
            confidence=confidence,
        )

    # --------------------------------------------------------------------------
    # Internal Methods
    # --------------------------------------------------------------------------

    @staticmethod
    def _calibrate_drift_var(median_ttf: float) -> float:
        """Calibrate drift variance for edge decay detector.

        Want: Sharpe decline 1.0→0.0 over median_ttf days should produce
        decay_probability > 0.7 within 0.7 * median_ttf days.

        Approximate: drift_var ≈ 1.0 / (median_ttf * 0.7)^2
        (so the cumulative drift signal reaches significance in time)
        """
        detection_window = median_ttf * 0.7
        if detection_window <= 0:
            return 0.01
        return min(0.1, max(0.001, 1.0 / (detection_window ** 2)))

    @staticmethod
    def _hartigan_dip_proxy(values: np.ndarray) -> Tuple[bool, float, float]:
        """Simplified Hartigan's dip test proxy.

        Uses histogram valley detection as a bimodality indicator.
        Two methods tried:
        1. Histogram peak-valley analysis
        2. Sorted-gap analysis (detects well-separated clusters)
        """
        if len(values) < 10:
            return False, 0.0, 1.0

        # Method 1: Sorted-gap analysis
        # If there's a large gap in the sorted values, it indicates bimodality
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        gaps = np.diff(sorted_vals)
        if len(gaps) > 2:
            # Find the largest gap relative to the data range
            data_range = sorted_vals[-1] - sorted_vals[0]
            if data_range > 0:
                max_gap = float(np.max(gaps))
                max_gap_idx = int(np.argmax(gaps))
                gap_ratio = max_gap / data_range

                # The gap should split data into two substantial groups
                left_frac = (max_gap_idx + 1) / n
                right_frac = 1.0 - left_frac
                min_frac = min(left_frac, right_frac)

                # Bimodal if largest gap is >25% of range AND each side has >15%
                if gap_ratio > 0.25 and min_frac > 0.15:
                    dip_stat = gap_ratio
                    p_value = max(0.01, 1.0 - dip_stat)
                    return True, float(dip_stat), p_value

        # Method 2: Histogram peak-valley analysis
        n_bins = min(20, n // 3)
        if n_bins < 5:
            return False, 0.0, 1.0

        counts, bin_edges = np.histogram(values, bins=n_bins)

        has_valley = False
        dip_stat = 0.0

        if len(counts) >= 3:
            max_count = max(counts)
            if max_count > 0:
                # Smooth counts slightly
                smoothed = counts.astype(float)
                for i in range(1, len(smoothed) - 1):
                    smoothed[i] = (counts[i - 1] + counts[i] + counts[i + 1]) / 3.0

                # Find peaks (local maxima) and also consider endpoints
                peaks = []
                if smoothed[0] > smoothed[1]:
                    peaks.append((0, smoothed[0]))
                for i in range(1, len(smoothed) - 1):
                    if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
                        peaks.append((i, smoothed[i]))
                if smoothed[-1] > smoothed[-2]:
                    peaks.append((len(smoothed) - 1, smoothed[-1]))

                # Look for valley between first two peaks
                if len(peaks) >= 2:
                    p1_idx = peaks[0][0]
                    p2_idx = peaks[1][0]
                    valley_region = smoothed[p1_idx:p2_idx + 1]
                    if len(valley_region) >= 3:
                        valley_min = float(min(valley_region))
                        peak_avg = (peaks[0][1] + peaks[1][1]) / 2
                        if peak_avg > 0:
                            dip_stat = 1.0 - valley_min / peak_avg
                            # Consider bimodal if valley is < 50% of peaks
                            has_valley = dip_stat > 0.5

        # Approximate p-value from dip statistic
        p_value = max(0.01, 1.0 - dip_stat) if has_valley else 1.0

        return bool(has_valley), float(dip_stat), p_value

    @staticmethod
    def _split_modes(values: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Split values into two modes at the optimal threshold.

        Uses the Otsu-like approach: find split that maximizes
        between-class variance.
        """
        if len(values) < 4:
            return None, None

        sorted_vals = np.sort(values)
        best_score = -1.0
        best_idx = len(sorted_vals) // 2

        # Try splits from 25% to 75%
        start = max(2, len(sorted_vals) // 4)
        end = min(len(sorted_vals) - 2, 3 * len(sorted_vals) // 4)

        for i in range(start, end):
            left = sorted_vals[:i]
            right = sorted_vals[i:]

            if len(left) < 2 or len(right) < 2:
                continue

            # Between-class variance
            n1 = len(left)
            n2 = len(right)
            m1 = left.mean()
            m2 = right.mean()
            n_total = n1 + n2
            score = (n1 * n2 / (n_total ** 2)) * (m1 - m2) ** 2

            if score > best_score:
                best_score = score
                best_idx = i

        short_mode = sorted_vals[:best_idx]
        long_mode = sorted_vals[best_idx:]

        return short_mode, long_mode
