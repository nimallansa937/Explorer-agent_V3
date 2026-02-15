"""
Tests for Phase 5: Production Feedback Loop

Covers all 18 required test areas from the v2.0 guide plus edge cases.

Explorer Prime v2.0 - Phase 5
"""

import math
import pytest
import numpy as np
from datetime import datetime, timedelta

from feedback.failure_archive import (
    FailureRecord,
    FailureArchive,
    FailureDistribution,
)
from feedback.structural_autopsy import (
    AutopsyResult,
    AntiTemplate,
    StructuralAutopsy,
    AntiTemplateInjector,
)
from feedback.meta_learning import (
    BimodalAnalysis,
    PipelineCalibration,
    MetaLearningSignal,
)


# ==============================================================================
# Test Helpers
# ==============================================================================

def _make_signal(n: int = 100, seed: int = 42) -> np.ndarray:
    """Create a synthetic trade signal history."""
    return np.random.RandomState(seed).randn(n)


def _make_failure(
    strategy_id: str = "strat_1",
    days_ago: int = 10,
    regime: str = "BULL",
    ttf: int = 30,
    signal_seed: int = 42,
    decay_type: str = "structural",
) -> FailureRecord:
    """Create a FailureRecord for testing."""
    return FailureRecord(
        strategy_id=strategy_id,
        failure_date=datetime.utcnow() - timedelta(days=days_ago),
        failure_regime=regime,
        decay_type=decay_type,
        trade_signal_history=_make_signal(100, signal_seed),
        time_to_failure_days=ttf,
    )


def _make_archive_with_bimodal(n_short: int = 20, n_long: int = 20) -> FailureArchive:
    """Create archive with bimodal TTF distribution."""
    archive = FailureArchive()
    rng = np.random.RandomState(42)

    # Short-lived failures (overfit): 10-20 days
    for i in range(n_short):
        archive.add(FailureRecord(
            strategy_id=f"short_{i}",
            failure_date=datetime.utcnow() - timedelta(days=rng.randint(1, 60)),
            failure_regime="BULL",
            time_to_failure_days=int(rng.normal(15, 3)),
            trade_signal_history=rng.randn(50),
            decay_type="structural",
        ))

    # Long-lived failures (edge decay): 80-100 days
    for i in range(n_long):
        archive.add(FailureRecord(
            strategy_id=f"long_{i}",
            failure_date=datetime.utcnow() - timedelta(days=rng.randint(1, 60)),
            failure_regime="BULL",
            time_to_failure_days=int(rng.normal(90, 5)),
            trade_signal_history=rng.randn(50),
            decay_type="feature",
        ))

    return archive


# ==============================================================================
# Test 1: Failure Archive — Behavioral Similarity
# ==============================================================================

class TestFailureArchiveBehavioralSimilarity:
    """test_failure_archive_behavioral_similarity: Uses trade signals, not feature vectors."""

    def test_high_behavioral_similarity_penalty(self):
        """Candidate with same signal should get high penalty."""
        archive = FailureArchive()
        signal = _make_signal(100, seed=42)
        archive.add(FailureRecord(
            strategy_id="failed_1",
            failure_date=datetime.utcnow(),
            failure_regime="BULL",
            trade_signal_history=signal.copy(),
        ))

        penalty = archive.penalty(signal, "BULL")
        assert penalty > 0.5  # High penalty for identical signal

    def test_different_signal_no_penalty(self):
        """Completely different signal should get no penalty."""
        archive = FailureArchive()
        archive.add(FailureRecord(
            strategy_id="failed_1",
            failure_date=datetime.utcnow(),
            failure_regime="BULL",
            trade_signal_history=_make_signal(100, seed=42),
        ))

        different_signal = _make_signal(100, seed=999)
        penalty = archive.penalty(different_signal, "BULL")
        # Should be 0 if correlation < 0.7
        assert penalty < 0.1


# ==============================================================================
# Test 2: Failure Archive — Time Decay
# ==============================================================================

class TestFailureArchiveTimeDecay:
    """test_failure_archive_time_decay: Penalty decays with 120-day half-life."""

    def test_recent_failure_high_penalty(self):
        """Recent failure should have higher penalty."""
        archive = FailureArchive()
        signal = _make_signal(100, seed=42)
        archive.add(FailureRecord(
            strategy_id="recent",
            failure_date=datetime.utcnow(),
            failure_regime="BULL",
            trade_signal_history=signal.copy(),
        ))

        penalty_recent = archive.penalty(signal, "BULL")
        assert penalty_recent > 0.5

    def test_old_failure_lower_penalty(self):
        """Old failure should have lower penalty than recent."""
        archive = FailureArchive()
        signal = _make_signal(100, seed=42)

        # Add old failure
        archive.add(FailureRecord(
            strategy_id="old",
            failure_date=datetime.utcnow() - timedelta(days=120),
            failure_regime="BULL",
            trade_signal_history=signal.copy(),
        ))

        ref_now = datetime.utcnow()
        penalty_old = archive.penalty(signal, "BULL", reference_date=ref_now)

        # Add recent failure
        archive2 = FailureArchive()
        archive2.add(FailureRecord(
            strategy_id="recent",
            failure_date=datetime.utcnow(),
            failure_regime="BULL",
            trade_signal_history=signal.copy(),
        ))
        penalty_recent = archive2.penalty(signal, "BULL", reference_date=ref_now)

        assert penalty_old < penalty_recent

    def test_120_day_half_life(self):
        """At 120 days, penalty should be roughly half."""
        archive = FailureArchive()
        signal = _make_signal(100, seed=42)
        now = datetime.utcnow()

        archive.add(FailureRecord(
            strategy_id="f1",
            failure_date=now,
            failure_regime="BULL",
            trade_signal_history=signal.copy(),
        ))
        p_now = archive.penalty(signal, "BULL", reference_date=now)

        p_120 = archive.penalty(signal, "BULL",
                                reference_date=now + timedelta(days=120))

        # Should be approximately half
        if p_now > 0:
            ratio = p_120 / p_now
            assert 0.4 <= ratio <= 0.6


# ==============================================================================
# Test 3: Failure Archive — Regime Amplifier
# ==============================================================================

class TestFailureArchiveRegimeAmplifier:
    """test_failure_archive_regime_amplifier: Same-regime penalties higher."""

    def test_same_regime_higher(self):
        archive = FailureArchive()
        signal = _make_signal(100, seed=42)
        archive.add(FailureRecord(
            strategy_id="f1",
            failure_date=datetime.utcnow(),
            failure_regime="BULL",
            trade_signal_history=signal.copy(),
        ))

        p_same = archive.penalty(signal, "BULL")
        p_diff = archive.penalty(signal, "BEAR")

        assert p_same > p_diff

    def test_different_regime_has_floor(self):
        """Different regime still has floor penalty (0.3)."""
        archive = FailureArchive()
        signal = _make_signal(100, seed=42)
        archive.add(FailureRecord(
            strategy_id="f1",
            failure_date=datetime.utcnow(),
            failure_regime="BULL",
            trade_signal_history=signal.copy(),
        ))

        p_diff = archive.penalty(signal, "BEAR")
        assert p_diff > 0  # Floor ensures non-zero penalty


# ==============================================================================
# Test 4: Failure Archive — Penalty Cap
# ==============================================================================

class TestFailureArchivePenaltyCap:
    """test_failure_archive_penalty_cap: Never exceeds 50% fitness reduction."""

    def test_max_reduction_50_percent(self):
        archive = FailureArchive()
        signal = _make_signal(100, seed=42)

        # Add many identical failures
        for i in range(20):
            archive.add(FailureRecord(
                strategy_id=f"f_{i}",
                failure_date=datetime.utcnow(),
                failure_regime="BULL",
                trade_signal_history=signal.copy(),
            ))

        penalty = archive.penalty(signal, "BULL")
        fitness = archive.apply_penalty(1.0, penalty)

        # Fitness should never go below 50% of original
        assert fitness >= 0.5

    def test_zero_penalty_no_reduction(self):
        archive = FailureArchive()
        fitness = archive.apply_penalty(1.0, 0.0)
        assert fitness == 1.0


# ==============================================================================
# Test 5: Failure Archive — Ring Buffer
# ==============================================================================

class TestFailureArchiveRingBuffer:
    """test_failure_archive_ring_buffer: Oldest records evicted at capacity."""

    def test_evicts_oldest(self):
        archive = FailureArchive(max_records=5)

        for i in range(7):
            archive.add(FailureRecord(
                strategy_id=f"strat_{i}",
                time_to_failure_days=i * 10,
            ))

        assert len(archive) == 5
        # Oldest (strat_0, strat_1) should be evicted
        ids = [r.strategy_id for r in archive.records]
        assert "strat_0" not in ids
        assert "strat_1" not in ids
        assert "strat_6" in ids


# ==============================================================================
# Test 6: Autopsy — Structural
# ==============================================================================

class TestAutopsyStructural:
    """test_autopsy_structural: Losing trades in existing features → anti-templates."""

    def test_structural_produces_anti_templates(self):
        autopsy = StructuralAutopsy()
        result = autopsy.analyze(
            strategy_id="failed_strat",
            genome=None,  # Will produce None encoding
            gap_type="structural",
        )
        assert result.gap_type == "structural"
        assert result.feature_investigation_priority == "HIGH"

    def test_structural_with_genome(self):
        """Genome with to_dict should produce anti-templates."""
        class MockGenome:
            def to_dict(self):
                return {"tree": [1, 2, 3], "mode": "flat"}

        autopsy = StructuralAutopsy()
        result = autopsy.analyze("s1", MockGenome(), gap_type="structural")
        assert result.anti_templates is not None
        assert len(result.anti_templates) > 0
        assert result.anti_templates[0].topology_encoding is not None


# ==============================================================================
# Test 7: Autopsy — Feature
# ==============================================================================

class TestAutopsyFeature:
    """test_autopsy_feature: Losing trades need new features → queue investigation."""

    def test_feature_no_anti_templates(self):
        autopsy = StructuralAutopsy()
        result = autopsy.analyze("s1", None, gap_type="feature")
        assert result.gap_type == "feature"
        assert result.anti_templates is None
        assert result.feature_investigation_priority == "HIGH"


# ==============================================================================
# Test 8: Anti-Template Penalizes Similar
# ==============================================================================

class TestAntiTemplatePenalizesSimilar:
    """test_anti_template_penalizes_similar: Similar topologies penalized."""

    def test_identical_encoding_penalized(self):
        injector = AntiTemplateInjector()
        encoding = np.random.RandomState(42).randn(64)

        injector.add_template(AntiTemplate(
            topology_encoding=encoding.copy(),
            created_at=datetime.utcnow(),
        ))

        penalty = injector.penalty(encoding)
        assert penalty > 0.8  # Same vector → cosine sim ≈ 1.0

    def test_different_encoding_not_penalized(self):
        injector = AntiTemplateInjector()
        injector.add_template(AntiTemplate(
            topology_encoding=np.random.RandomState(42).randn(64),
            created_at=datetime.utcnow(),
        ))

        different = np.random.RandomState(999).randn(64)
        penalty = injector.penalty(different)
        # Different random vector → cosine sim near 0
        assert penalty < 0.1


# ==============================================================================
# Test 9: Anti-Template Decay
# ==============================================================================

class TestAntiTemplateDecay:
    """test_anti_template_decay: 90-day half-life on anti-templates."""

    def test_decay_over_time(self):
        injector = AntiTemplateInjector()
        encoding = np.random.RandomState(42).randn(64)
        now = datetime.utcnow()

        injector.add_template(AntiTemplate(
            topology_encoding=encoding.copy(),
            created_at=now,
            half_life_days=90.0,
        ))

        p_now = injector.penalty(encoding, reference_date=now)
        p_90 = injector.penalty(encoding, reference_date=now + timedelta(days=90))

        if p_now > 0:
            ratio = p_90 / p_now
            assert 0.4 <= ratio <= 0.6  # Half-life check

    def test_prune_expired(self):
        injector = AntiTemplateInjector()
        old = datetime.utcnow() - timedelta(days=1000)  # Very old

        injector.add_template(AntiTemplate(
            topology_encoding=np.ones(64),
            created_at=old,
            half_life_days=90.0,
        ))

        removed = injector.prune_expired()
        assert removed == 1
        assert len(injector) == 0


# ==============================================================================
# Test 10: Meta — Decay Timescale
# ==============================================================================

class TestMetaDecayTimescale:
    """test_meta_decay_timescale: Correct median computed from distribution."""

    def test_median_ttf(self):
        archive = FailureArchive()
        # Need >= MIN_RECORDS_FOR_ANALYSIS (10) records
        # Add records with TTF: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        for ttf in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            archive.add(FailureRecord(
                strategy_id=f"s_{ttf}",
                time_to_failure_days=ttf,
            ))

        meta = MetaLearningSignal(archive)
        timescale = meta.compute_characteristic_decay_timescale()
        # Integer median of sorted [10,20,30,40,50,60,70,80,90,100]
        # Index 5 (len//2) = 60
        assert timescale == 60.0

    def test_too_few_records_default(self):
        archive = FailureArchive()
        archive.add(FailureRecord(strategy_id="s1", time_to_failure_days=30))

        meta = MetaLearningSignal(archive)
        timescale = meta.compute_characteristic_decay_timescale()
        assert timescale == 50.0  # Default


# ==============================================================================
# Test 11: Meta — Calibrates Shadow Duration
# ==============================================================================

class TestMetaCalibratesShadowDuration:
    """test_meta_calibrates_shadow_duration: Shadow duration adjusts proportionally."""

    def test_short_timescale_reduces_shadow(self):
        archive = FailureArchive()
        # Median TTF = 30 → shadow = min(14, 30 * 0.3) = min(14, 9) = 9
        for ttf in [20, 25, 30, 35, 40]:
            archive.add(FailureRecord(strategy_id=f"s_{ttf}", time_to_failure_days=ttf))
        for ttf in [22, 27, 32, 37, 42]:
            archive.add(FailureRecord(strategy_id=f"s2_{ttf}", time_to_failure_days=ttf))

        meta = MetaLearningSignal(archive)
        cal = meta.get_pipeline_calibration()
        assert cal.shadow_min_duration < 14

    def test_long_timescale_keeps_14(self):
        archive = FailureArchive()
        # Median TTF = 100 → shadow = min(14, 100 * 0.3) = min(14, 30) = 14
        for ttf in [80, 90, 100, 110, 120, 85, 95, 105, 115, 125]:
            archive.add(FailureRecord(strategy_id=f"s_{ttf}", time_to_failure_days=ttf))

        meta = MetaLearningSignal(archive)
        cal = meta.get_pipeline_calibration()
        assert cal.shadow_min_duration == 14


# ==============================================================================
# Test 12: Meta — Calibrates Drift Var
# ==============================================================================

class TestMetaCalibratesDriftVar:
    """test_meta_calibrates_drift_var: Drift variance matches decay timescale."""

    def test_drift_var_inversely_proportional(self):
        """Longer timescale → smaller drift_var."""
        # Short timescale
        archive_short = FailureArchive()
        for ttf in [10, 15, 20, 25, 30, 12, 17, 22, 27, 32]:
            archive_short.add(FailureRecord(strategy_id=f"s_{ttf}", time_to_failure_days=ttf))

        # Long timescale
        archive_long = FailureArchive()
        for ttf in [80, 90, 100, 110, 120, 85, 95, 105, 115, 125]:
            archive_long.add(FailureRecord(strategy_id=f"s_{ttf}", time_to_failure_days=ttf))

        cal_short = MetaLearningSignal(archive_short).get_pipeline_calibration()
        cal_long = MetaLearningSignal(archive_long).get_pipeline_calibration()

        assert cal_short.drift_var > cal_long.drift_var

    def test_drift_var_bounded(self):
        archive = FailureArchive()
        for ttf in [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]:
            archive.add(FailureRecord(strategy_id=f"s_{ttf}", time_to_failure_days=ttf))

        cal = MetaLearningSignal(archive).get_pipeline_calibration()
        assert 0.001 <= cal.drift_var <= 0.1


# ==============================================================================
# Test 13: Meta — Bimodal Detection
# ==============================================================================

class TestMetaBimodalDetection:
    """test_meta_bimodal_detection: Hartigan dip test identifies bimodality."""

    def test_bimodal_distribution_detected(self):
        archive = _make_archive_with_bimodal(30, 30)
        meta = MetaLearningSignal(archive)
        result = meta.detect_bimodal_failure()

        assert result is not None
        assert result.is_bimodal == True
        assert result.short_mode_median is not None
        assert result.long_mode_median is not None
        assert result.short_mode_median < result.long_mode_median

    def test_unimodal_not_detected(self):
        archive = FailureArchive()
        rng = np.random.RandomState(42)
        # Unimodal: all clustered around 50 days
        for i in range(40):
            archive.add(FailureRecord(
                strategy_id=f"s_{i}",
                time_to_failure_days=int(rng.normal(50, 5)),
            ))

        meta = MetaLearningSignal(archive)
        result = meta.detect_bimodal_failure()
        # Unimodal should not be detected as bimodal (or result is None)
        assert result is None or result.is_bimodal == False


# ==============================================================================
# Test 14: Meta — Bimodal Recommendations
# ==============================================================================

class TestMetaBimodalRecommendations:
    """test_meta_bimodal_recommendations: Correct gate adjustments for each mode."""

    def test_bimodal_has_recommendations(self):
        archive = _make_archive_with_bimodal(30, 30)
        meta = MetaLearningSignal(archive)
        result = meta.detect_bimodal_failure()

        if result and result.is_bimodal:
            assert result.recommended_gate_adjustments is not None
            assert "tighten_validation" in result.recommended_gate_adjustments
            assert "advice" in result.recommended_gate_adjustments


# ==============================================================================
# Test 15: Pipeline Calibration Aggregates
# ==============================================================================

class TestPipelineCalibrationAggregates:
    """test_pipeline_calibration_aggregates: All signals combined into one recommendation."""

    def test_calibration_complete(self):
        archive = FailureArchive()
        for i, ttf in enumerate([20, 30, 40, 50, 60, 25, 35, 45, 55, 65]):
            archive.add(FailureRecord(strategy_id=f"s_{i}", time_to_failure_days=ttf))

        meta = MetaLearningSignal(archive)
        cal = meta.get_pipeline_calibration()

        assert cal.shadow_min_duration > 0
        assert cal.drift_var > 0
        assert cal.archive_half_life > 0
        assert cal.characteristic_decay_timescale > 0
        assert 0 <= cal.confidence <= 1.0

    def test_archive_half_life_proportional(self):
        """Archive half-life should be ~2x median TTF."""
        archive = FailureArchive()
        for i, ttf in enumerate([40, 45, 50, 55, 60, 42, 47, 52, 57, 62]):
            archive.add(FailureRecord(strategy_id=f"s_{i}", time_to_failure_days=ttf))

        meta = MetaLearningSignal(archive)
        cal = meta.get_pipeline_calibration()

        # median_ttf ≈ 51, so archive_half_life ≈ 102
        assert cal.archive_half_life >= cal.characteristic_decay_timescale * 1.5


# ==============================================================================
# Test 16: Feedback Loop — Structural Path
# ==============================================================================

class TestFeedbackLoopStructural:
    """test_feedback_loop_structural: Failure → autopsy → anti-template → generation avoids."""

    def test_structural_feedback_path(self):
        """Complete structural feedback loop."""
        # 1. Strategy fails
        class MockGenome:
            def to_dict(self):
                return {"tree": [1, 2, 3]}

        # 2. Autopsy extracts anti-template
        autopsy = StructuralAutopsy()
        result = autopsy.analyze("s1", MockGenome(), gap_type="structural")
        assert result.anti_templates is not None

        # 3. Anti-template injector penalizes similar
        injector = AntiTemplateInjector()
        injector.add_from_autopsy(result)
        assert len(injector) > 0

        # 4. Similar candidate gets penalized
        encoding = result.anti_templates[0].topology_encoding
        penalty = injector.penalty(encoding)
        assert penalty > 0


# ==============================================================================
# Test 17: Feedback Loop — Feature Path
# ==============================================================================

class TestFeedbackLoopFeature:
    """test_feedback_loop_feature: Failure → autopsy → feature scout → investigation queued."""

    def test_feature_feedback_path(self):
        """Feature decay queues investigation."""
        autopsy = StructuralAutopsy()
        result = autopsy.analyze("s1", None, gap_type="feature")

        assert result.gap_type == "feature"
        assert result.feature_investigation_priority == "HIGH"
        assert result.anti_templates is None


# ==============================================================================
# Test 18: Full Loop Closes
# ==============================================================================

class TestFullLoopCloses:
    """test_full_loop_closes: Production failure information reaches generation layer."""

    def test_complete_loop(self):
        """Failure → archive → penalty on new candidate → generation affected."""
        # 1. Production failure
        archive = FailureArchive()
        signal = _make_signal(100, seed=42)
        archive.add(FailureRecord(
            strategy_id="prod_failure",
            failure_date=datetime.utcnow(),
            failure_regime="BULL",
            trade_signal_history=signal.copy(),
            time_to_failure_days=30,
        ))

        # 2. Penalty affects candidate evaluation
        penalty = archive.penalty(signal, "BULL")
        original_fitness = 1.0
        adjusted_fitness = archive.apply_penalty(original_fitness, penalty)

        # 3. Generation layer sees reduced fitness
        assert adjusted_fitness < original_fitness

        # 4. Meta-learning calibrates pipeline
        for i in range(15):
            archive.add(FailureRecord(
                strategy_id=f"s_{i}",
                time_to_failure_days=30 + i * 5,
                trade_signal_history=np.random.RandomState(i).randn(50),
            ))

        meta = MetaLearningSignal(archive)
        cal = meta.get_pipeline_calibration()
        assert cal.shadow_min_duration > 0
        assert cal.confidence > 0


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Additional edge case tests."""

    def test_empty_archive_no_penalty(self):
        archive = FailureArchive()
        signal = _make_signal(100)
        assert archive.penalty(signal, "BULL") == 0.0

    def test_none_signal_no_penalty(self):
        archive = FailureArchive()
        archive.add(_make_failure())
        assert archive.penalty(None, "BULL") == 0.0

    def test_anti_template_injector_empty(self):
        injector = AntiTemplateInjector()
        assert injector.penalty(np.ones(64)) == 0.0

    def test_failure_distribution_empty(self):
        archive = FailureArchive()
        dist = archive.get_failure_distribution()
        assert dist.n_records == 0
        assert dist.median_time_to_failure == 0.0

    def test_meta_learning_insufficient_records(self):
        archive = FailureArchive()
        archive.add(FailureRecord(strategy_id="s1", time_to_failure_days=30))
        meta = MetaLearningSignal(archive)
        result = meta.detect_bimodal_failure()
        assert result is None

    def test_calibration_confidence_scales(self):
        archive = FailureArchive()
        for i in range(50):
            archive.add(FailureRecord(
                strategy_id=f"s_{i}", time_to_failure_days=30 + i,
            ))
        cal = MetaLearningSignal(archive).get_pipeline_calibration()
        assert cal.confidence == 0.5  # 50/100

        for i in range(50, 100):
            archive.add(FailureRecord(
                strategy_id=f"s_{i}", time_to_failure_days=30 + i,
            ))
        cal2 = MetaLearningSignal(archive).get_pipeline_calibration()
        assert cal2.confidence == 1.0  # 100/100
