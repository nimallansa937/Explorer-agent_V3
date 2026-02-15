"""
Tests for Phase 4: Sequential Intervention Protocol

Covers all 12 required test areas from the v2.0 guide plus edge cases.

Explorer Prime v2.0 - Phase 4
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from diagnostics.anomaly_signature import (
    AnomalySignature,
    GapType,
    GapClassification,
    DiagnosisResult,
    TemporalProfile,
    RegimeProfile,
    LeadLagProfile,
    TradeOpportunity,
)
from diagnostics.intervention_protocol import (
    InterventionType,
    PlanStatus,
    Phase1Decision,
    InterventionPlan,
    Phase1Result,
    FeatureProposal,
    TreeSeed,
    InterventionRouter,
    DirectedFeatureScout,
    StructuralSeedInjector,
)
from diagnostics.anomaly_diagnostic import SimpleRandomForest


# ==============================================================================
# Test Helpers
# ==============================================================================

def _make_diagnosis(core_overlap: float, gap_type: Optional[GapType] = None,
                    temporal_pattern: bool = False,
                    transition_concentrated: bool = False,
                    has_leads: bool = False) -> DiagnosisResult:
    """Create a DiagnosisResult with specified core_overlap."""
    if gap_type is None:
        if core_overlap >= 0.7:
            gap_type = GapType.STRUCTURAL
        elif core_overlap <= 0.4:
            gap_type = GapType.FEATURE
        else:
            gap_type = GapType.AMBIGUOUS

    sig = AnomalySignature()
    sig.temporal_clustering.has_strong_pattern = temporal_pattern
    sig.temporal_clustering.peak_hours = [9, 10] if temporal_pattern else []
    sig.regime_distribution.concentrated_at_transitions = transition_concentrated
    sig.regime_distribution.transition_types = [("BULL", "BEAR")] if transition_concentrated else []
    sig.regime_distribution.transition_fraction = 0.6 if transition_concentrated else 0.0
    sig.lead_lag_structure.has_significant_leads = has_leads
    sig.lead_lag_structure.leading_instruments = ["GOLD", "EUR"] if has_leads else []
    sig.lead_lag_structure.cross_asset_correlations = {"GOLD": 0.5, "EUR": 0.4} if has_leads else {}

    classification = GapClassification(
        core_overlap=core_overlap,
        gap_type=gap_type,
        top_features=[("f1", 0.3), ("f2", 0.2)],
        baseline_auc=0.75,
        missing_features=["f_missing_1"] if gap_type != GapType.STRUCTURAL else [],
    )

    action = {
        GapType.STRUCTURAL: "structural_seeds",
        GapType.FEATURE: "feature_scout",
        GapType.AMBIGUOUS: "sequential_intervention",
    }.get(gap_type, "sequential_intervention")

    return DiagnosisResult(
        signature=sig,
        classification=classification,
        recommended_action=action,
        confidence=0.75,
    )


# ==============================================================================
# Test 1: Route Structural
# ==============================================================================

class TestRouteStructural:
    """test_route_structural: High overlap → structural only."""

    def test_high_overlap_structural(self):
        diagnosis = _make_diagnosis(core_overlap=0.8)
        router = InterventionRouter()
        plan = router.route(diagnosis)

        assert plan.phase_1 == InterventionType.STRUCTURAL_SEEDS
        assert plan.phase_2 is None
        assert plan.attribution_window_days == 0
        assert plan.confounded_tag is False

    def test_exactly_0_7_structural(self):
        diagnosis = _make_diagnosis(core_overlap=0.7)
        router = InterventionRouter()
        plan = router.route(diagnosis)
        assert plan.phase_1 == InterventionType.STRUCTURAL_SEEDS
        assert plan.phase_2 is None


# ==============================================================================
# Test 2: Route Feature
# ==============================================================================

class TestRouteFeature:
    """test_route_feature: Low overlap → feature scout only."""

    def test_low_overlap_feature(self):
        diagnosis = _make_diagnosis(core_overlap=0.2)
        router = InterventionRouter()
        plan = router.route(diagnosis)

        assert plan.phase_1 == InterventionType.FEATURE_SCOUT
        assert plan.phase_2 is None
        assert plan.attribution_window_days == 0

    def test_exactly_0_4_feature(self):
        diagnosis = _make_diagnosis(core_overlap=0.4)
        router = InterventionRouter()
        plan = router.route(diagnosis)
        assert plan.phase_1 == InterventionType.FEATURE_SCOUT
        assert plan.phase_2 is None


# ==============================================================================
# Test 3: Route Ambiguous Sequential
# ==============================================================================

class TestRouteAmbiguousSequential:
    """test_route_ambiguous_sequential: Mid overlap → sequential protocol."""

    def test_ambiguous_sequential(self):
        diagnosis = _make_diagnosis(core_overlap=0.55)
        router = InterventionRouter()
        plan = router.route(diagnosis)

        assert plan.phase_1 == InterventionType.STRUCTURAL_SEEDS
        assert plan.phase_2 == InterventionType.CONDITIONAL_FEATURE_SCOUT
        assert plan.attribution_window_days == 45
        assert plan.confounded_tag is False

    def test_ambiguous_at_0_5(self):
        diagnosis = _make_diagnosis(core_overlap=0.5)
        router = InterventionRouter()
        plan = router.route(diagnosis)
        assert plan.phase_2 == InterventionType.CONDITIONAL_FEATURE_SCOUT
        assert plan.attribution_window_days == 45

    def test_plan_anomaly_id_linked(self):
        diagnosis = _make_diagnosis(core_overlap=0.55)
        router = InterventionRouter()
        plan = router.route(diagnosis)
        assert plan.anomaly_id == diagnosis.signature.anomaly_id


# ==============================================================================
# Test 4: Phase 1 Sufficient (>70%)
# ==============================================================================

class TestPhase1Sufficient:
    """test_phase_1_sufficient: >70% resolution → CLOSE."""

    def test_high_resolution_closes(self):
        router = InterventionRouter()
        plan = InterventionPlan(
            phase_1=InterventionType.STRUCTURAL_SEEDS,
            phase_2=InterventionType.CONDITIONAL_FEATURE_SCOUT,
            attribution_window_days=45,
        )

        result = router.evaluate_phase_1(
            plan,
            original_anomaly_rate=0.10,
            current_anomaly_rate=0.02,  # 80% resolution
        )

        assert result.resolution_rate == pytest.approx(0.8, abs=0.01)
        assert result.decision == Phase1Decision.CLOSE
        assert result.confounded is False

    def test_71_percent_closes(self):
        router = InterventionRouter()
        plan = InterventionPlan()
        result = router.evaluate_phase_1(plan, 1.0, 0.29)
        assert result.decision == Phase1Decision.CLOSE


# ==============================================================================
# Test 5: Phase 1 Insufficient (<30%)
# ==============================================================================

class TestPhase1Insufficient:
    """test_phase_1_insufficient: <30% resolution → PROCEED_TO_FEATURE."""

    def test_low_resolution_proceeds(self):
        router = InterventionRouter()
        plan = InterventionPlan()

        result = router.evaluate_phase_1(
            plan,
            original_anomaly_rate=0.10,
            current_anomaly_rate=0.08,  # 20% resolution
        )

        assert result.resolution_rate == pytest.approx(0.2, abs=0.01)
        assert result.decision == Phase1Decision.PROCEED_TO_FEATURE
        assert result.confounded is False

    def test_zero_resolution_proceeds(self):
        router = InterventionRouter()
        plan = InterventionPlan()
        result = router.evaluate_phase_1(plan, 0.10, 0.10)
        assert result.decision == Phase1Decision.PROCEED_TO_FEATURE


# ==============================================================================
# Test 6: Phase 1 Mixed (30-70%)
# ==============================================================================

class TestPhase1Mixed:
    """test_phase_1_mixed: 30-70% → PROCEED_CONFOUNDED with tag."""

    def test_mixed_resolution_confounded(self):
        router = InterventionRouter()
        plan = InterventionPlan()

        result = router.evaluate_phase_1(
            plan,
            original_anomaly_rate=0.10,
            current_anomaly_rate=0.05,  # 50% resolution
        )

        assert result.resolution_rate == pytest.approx(0.5, abs=0.01)
        assert result.decision == Phase1Decision.PROCEED_CONFOUNDED
        assert result.confounded is True
        assert plan.confounded_tag is True

    def test_30_percent_confounded(self):
        router = InterventionRouter()
        plan = InterventionPlan()
        result = router.evaluate_phase_1(plan, 1.0, 0.70)
        assert result.decision == Phase1Decision.PROCEED_CONFOUNDED


# ==============================================================================
# Test 7: Confounded Tag Raises Threshold
# ==============================================================================

class TestConfoundedTagRaisesThreshold:
    """test_confounded_tag_raises_threshold: Promotion requires 1.5x not 1.0x."""

    def test_confounded_multiplier(self):
        router = InterventionRouter()
        plan = InterventionPlan(confounded_tag=True)
        multiplier = router.get_confounded_promotion_multiplier(plan)
        assert multiplier == 1.5

    def test_clean_multiplier(self):
        router = InterventionRouter()
        plan = InterventionPlan(confounded_tag=False)
        multiplier = router.get_confounded_promotion_multiplier(plan)
        assert multiplier == 1.0

    def test_confounded_set_by_evaluate(self):
        router = InterventionRouter()
        plan = InterventionPlan()
        assert plan.confounded_tag is False

        # 50% resolution → confounded
        router.evaluate_phase_1(plan, 0.10, 0.05)
        assert plan.confounded_tag is True
        assert router.get_confounded_promotion_multiplier(plan) == 1.5


# ==============================================================================
# Test 8: Regime Stability Pauses Window
# ==============================================================================

class TestRegimeStabilityPausesWindow:
    """test_regime_stability_pauses_window: Regime change pauses attribution clock."""

    def test_same_regime_active(self):
        router = InterventionRouter()
        plan = InterventionPlan(regime_at_start="BULL")
        is_active = router.check_regime_stability(plan, "BULL", "BULL")
        assert is_active is True
        assert plan.paused_days == 0

    def test_regime_change_pauses(self):
        router = InterventionRouter()
        plan = InterventionPlan(regime_at_start="BULL")
        is_active = router.check_regime_stability(plan, "BULL", "BEAR")
        assert is_active is False
        assert plan.paused_days == 1

    def test_multiple_pauses_accumulate(self):
        router = InterventionRouter()
        plan = InterventionPlan(regime_at_start="BULL")

        router.check_regime_stability(plan, "BULL", "BEAR")
        router.check_regime_stability(plan, "BULL", "RANGE")
        router.check_regime_stability(plan, "BULL", "BEAR")

        assert plan.paused_days == 3
        assert len(plan.regime_transitions) == 3

    def test_regime_stability_disabled(self):
        router = InterventionRouter()
        router.REGIME_STABILITY_REQUIRED = False
        plan = InterventionPlan()
        is_active = router.check_regime_stability(plan, "BULL", "BEAR")
        assert is_active is True

    def test_effective_attribution_days(self):
        plan = InterventionPlan(attribution_window_days=45)
        plan.paused_days = 10
        assert plan.effective_attribution_days() == 35


# ==============================================================================
# Test 9: Directed Scout — Temporal
# ==============================================================================

class TestDirectedScoutTemporal:
    """test_directed_scout_temporal: Temporal clustering → event search."""

    def test_temporal_pattern_triggers_event_search(self):
        diagnosis = _make_diagnosis(
            core_overlap=0.3, temporal_pattern=True
        )
        scout = DirectedFeatureScout()
        proposals = scout.search_from_signature(diagnosis.signature)

        assert len(proposals) > 0
        event_proposals = [p for p in proposals if p.source_channel == "event_driven"]
        assert len(event_proposals) > 0

    def test_no_temporal_no_event(self):
        diagnosis = _make_diagnosis(core_overlap=0.3, temporal_pattern=False)
        scout = DirectedFeatureScout()
        proposals = scout.search_from_signature(diagnosis.signature)
        event_proposals = [p for p in proposals if p.source_channel == "event_driven"]
        assert len(event_proposals) == 0

    def test_proposals_have_required_fields(self):
        diagnosis = _make_diagnosis(core_overlap=0.3, temporal_pattern=True)
        scout = DirectedFeatureScout()
        proposals = scout.search_from_signature(diagnosis.signature)
        for p in proposals:
            assert p.feature_id
            assert p.source_channel
            assert 0 <= p.relevance_score <= 1.0
            assert p.compute_specification


# ==============================================================================
# Test 10: Directed Scout — Regime
# ==============================================================================

class TestDirectedScoutRegime:
    """test_directed_scout_regime: Transition clustering → regime search."""

    def test_transition_triggers_regime_search(self):
        diagnosis = _make_diagnosis(
            core_overlap=0.3, transition_concentrated=True
        )
        scout = DirectedFeatureScout()
        proposals = scout.search_from_signature(diagnosis.signature)

        regime_proposals = [p for p in proposals if p.source_channel == "regime_transition"]
        assert len(regime_proposals) > 0

    def test_no_transitions_no_regime(self):
        diagnosis = _make_diagnosis(core_overlap=0.3, transition_concentrated=False)
        scout = DirectedFeatureScout()
        proposals = scout.search_from_signature(diagnosis.signature)
        regime_proposals = [p for p in proposals if p.source_channel == "regime_transition"]
        assert len(regime_proposals) == 0


# ==============================================================================
# Test 11: Directed Scout — Cross-Asset
# ==============================================================================

class TestDirectedScoutCrossasset:
    """test_directed_scout_crossasset: Lead-lag → cross-asset search."""

    def test_leads_trigger_crossasset_search(self):
        diagnosis = _make_diagnosis(core_overlap=0.3, has_leads=True)
        scout = DirectedFeatureScout()
        proposals = scout.search_from_signature(diagnosis.signature)

        ca_proposals = [p for p in proposals if p.source_channel == "cross_asset"]
        assert len(ca_proposals) > 0
        # Should have proposals for GOLD and EUR
        feature_ids = {p.feature_id for p in ca_proposals}
        assert any("GOLD" in fid for fid in feature_ids)
        assert any("EUR" in fid for fid in feature_ids)

    def test_no_leads_no_crossasset(self):
        diagnosis = _make_diagnosis(core_overlap=0.3, has_leads=False)
        scout = DirectedFeatureScout()
        proposals = scout.search_from_signature(diagnosis.signature)
        ca_proposals = [p for p in proposals if p.source_channel == "cross_asset"]
        assert len(ca_proposals) == 0


# ==============================================================================
# Test 12: Validate Fill — AUC Threshold
# ==============================================================================

class TestValidateFillAucThreshold:
    """test_validate_fill_auc_threshold: AUC improvement >= 0.08 required."""

    def test_good_feature_passes(self):
        """A truly informative feature should pass the AUC threshold."""
        rng = np.random.RandomState(42)
        n = 100
        missed = rng.randn(n, 5)
        control = rng.randn(n, 5)

        # Proposed feature: perfectly separates missed from control
        proposed = np.concatenate([np.ones(n), np.zeros(n)])

        scout = DirectedFeatureScout()
        result = scout.validate_fill(proposed, missed, control, baseline_auc=0.5)
        # Should pass (AUC with perfect feature should be much better)
        assert result == True

    def test_useless_feature_fails(self):
        """A random feature should fail the AUC threshold."""
        rng = np.random.RandomState(42)
        n = 100
        missed = rng.randn(n, 5)
        control = rng.randn(n, 5)

        # Proposed feature: pure noise
        proposed = rng.randn(n * 2)

        scout = DirectedFeatureScout()
        result = scout.validate_fill(proposed, missed, control, baseline_auc=0.95)
        # Should fail (noise doesn't improve AUC above already high baseline)
        assert result == False

    def test_threshold_is_008(self):
        """Verify the threshold constant."""
        scout = DirectedFeatureScout()
        assert scout.AUC_IMPROVEMENT_THRESHOLD == 0.08


# ==============================================================================
# Additional Tests
# ==============================================================================

class TestPlanLifecycle:
    """Tests for InterventionPlan lifecycle management."""

    def test_full_lifecycle(self):
        plan = InterventionPlan(
            phase_1=InterventionType.STRUCTURAL_SEEDS,
            phase_2=InterventionType.CONDITIONAL_FEATURE_SCOUT,
            attribution_window_days=45,
        )

        assert plan.status == PlanStatus.PENDING

        plan.start_phase_1(regime="BULL")
        assert plan.status == PlanStatus.PHASE_1_ACTIVE
        assert plan.regime_at_start == "BULL"

        plan.complete_phase_1()
        assert plan.status == PlanStatus.AWAITING_ATTRIBUTION

        plan.start_phase_2()
        assert plan.status == PlanStatus.PHASE_2_ACTIVE

        plan.complete()
        assert plan.status == PlanStatus.COMPLETED

    def test_single_phase_lifecycle(self):
        plan = InterventionPlan(
            phase_1=InterventionType.STRUCTURAL_SEEDS,
            phase_2=None,
        )

        plan.start_phase_1()
        plan.complete_phase_1()
        assert plan.status == PlanStatus.COMPLETED

    def test_cancel(self):
        plan = InterventionPlan()
        plan.start_phase_1()
        plan.cancel()
        assert plan.status == PlanStatus.CANCELLED


class TestStructuralSeedInjector:
    """Tests for StructuralSeedInjector."""

    def test_extract_seeds(self):
        """Seeds should be extractable from RF."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = (X[:, 0] > 0).astype(float)
        rf = SimpleRandomForest(n_estimators=10, max_depth=3, random_state=42)
        rf.fit(X, y)

        injector = StructuralSeedInjector()
        seeds = injector.extract_tree_seeds(rf, source_anomaly_id="test-123")

        assert len(seeds) > 0
        for seed in seeds:
            assert seed.source_anomaly_id == "test-123"
            assert len(seed.feature_splits) > 0
            assert seed.depth > 0

    def test_inject_seeds_fraction(self):
        """Seeds should replace 20% of population."""
        injector = StructuralSeedInjector()
        seeds = [TreeSeed() for _ in range(10)]
        spec = injector.inject_seeds(seeds, population_size=100)

        assert spec["n_seed_slots"] == 20
        assert spec["n_seeds_available"] <= 10
        assert spec["replacement_fraction"] == 0.20

    def test_inject_fewer_seeds_than_slots(self):
        """When fewer seeds than slots, use all available."""
        injector = StructuralSeedInjector()
        seeds = [TreeSeed() for _ in range(3)]
        spec = injector.inject_seeds(seeds, population_size=100)
        assert spec["n_seeds_available"] == 3


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_original_anomaly_rate(self):
        router = InterventionRouter()
        plan = InterventionPlan()
        result = router.evaluate_phase_1(plan, 0.0, 0.0)
        assert result.decision == Phase1Decision.CLOSE

    def test_negative_resolution_clamped(self):
        router = InterventionRouter()
        plan = InterventionPlan()
        # Current worse than original
        result = router.evaluate_phase_1(plan, 0.05, 0.10)
        assert result.resolution_rate == 0.0

    def test_all_channels_combined(self):
        """Signature with all patterns triggers all search channels."""
        diagnosis = _make_diagnosis(
            core_overlap=0.3,
            temporal_pattern=True,
            transition_concentrated=True,
            has_leads=True,
        )
        scout = DirectedFeatureScout()
        proposals = scout.search_from_signature(diagnosis.signature)

        channels = {p.source_channel for p in proposals}
        assert "event_driven" in channels
        assert "regime_transition" in channels
        assert "cross_asset" in channels

    def test_empty_signature_no_proposals(self):
        """Empty signature should produce no proposals."""
        sig = AnomalySignature()
        scout = DirectedFeatureScout()
        proposals = scout.search_from_signature(sig)
        assert len(proposals) == 0

    def test_plan_ids_unique(self):
        p1 = InterventionPlan()
        p2 = InterventionPlan()
        assert p1.plan_id != p2.plan_id
