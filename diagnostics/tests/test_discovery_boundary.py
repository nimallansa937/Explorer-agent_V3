"""
Tests for Phase 8: Discovery Boundary Formalization

Covers all 10 required tests from the v2.0 guide plus edge cases.

Explorer Prime v2.0 - Phase 8
"""

import pytest
from diagnostics.discovery_boundary import (
    DiscoveryLevel,
    CapabilityCategory,
    DiscoveryBoundary,
    ComputationalLibrary,
    TransformSpec,
    DataStreamSpec,
    DiscoveryClassification,
    BoundarySurface,
    BoundaryExpansion,
    ResearchBrief,
    FeatureProposal,
    LEVEL_CAPABILITY_MAP,
)


# ==============================================================================
# Test Helpers
# ==============================================================================

def _make_library() -> ComputationalLibrary:
    """Create a library with standard transforms and data streams."""
    lib = ComputationalLibrary()

    # Add standard transforms
    for name, cat in [
        ("rolling_mean", "rolling_window_stats"),
        ("rolling_std", "rolling_window_stats"),
        ("rsi", "momentum_indicators"),
        ("macd", "momentum_indicators"),
        ("bollinger", "momentum_indicators"),
        ("ofi", "order_flow_metrics"),
        ("realized_vol", "volatility_estimators"),
        ("garman_klass", "volatility_estimators"),
        ("hmm_regime", "regime_detectors"),
    ]:
        lib.add_transform(TransformSpec(name=name, category=cat))

    # Add standard data streams
    for name, cat in [
        ("close_price", "price"),
        ("volume", "volume"),
        ("bid_ask_spread", "order_flow"),
        ("trade_flow", "order_flow"),
    ]:
        lib.add_data_stream(DataStreamSpec(name=name, category=cat))

    return lib


def _make_boundary() -> DiscoveryBoundary:
    """Create a DiscoveryBoundary with standard library."""
    return DiscoveryBoundary(computational_library=_make_library())


# ==============================================================================
# Test 1: Classify Recombination
# ==============================================================================

class TestClassifyRecombination:
    """test_classify_recombination: Existing streams + existing transforms → Level 1."""

    def test_recombination_level(self):
        boundary = _make_boundary()
        proposal = FeatureProposal(
            feature_id="rsi_close",
            data_streams_required=["close_price"],
            transforms_required=["rsi"],
        )

        result = boundary.classify_feature(proposal)
        assert result.level == DiscoveryLevel.RECOMBINATION
        assert result.capability == CapabilityCategory.AUTONOMOUS

    def test_recombination_has_autonomous_actions(self):
        boundary = _make_boundary()
        proposal = FeatureProposal(
            data_streams_required=["close_price", "volume"],
            transforms_required=["rolling_mean", "rolling_std"],
        )

        result = boundary.classify_feature(proposal)
        assert len(result.autonomous_actions) > 0
        assert len(result.human_actions) == 0


# ==============================================================================
# Test 2: Classify Timescale
# ==============================================================================

class TestClassifyTimescale:
    """test_classify_timescale: Existing stream at new timescale → Level 2."""

    def test_new_timescale_with_sweep(self):
        boundary = _make_boundary()
        proposal = FeatureProposal(
            data_streams_required=["close_price"],
            transforms_required=["rolling_mean"],
            timescale="30sec",  # Not in default timescales
        )

        result = boundary.classify_feature(proposal)
        assert result.level == DiscoveryLevel.TIMESCALE
        assert result.capability == CapabilityCategory.AUTONOMOUS

    def test_new_timescale_without_sweep(self):
        boundary = _make_boundary()
        boundary.library.timescale_sweep_configured = False

        proposal = FeatureProposal(
            data_streams_required=["close_price"],
            transforms_required=["rsi"],
            timescale="30sec",
        )

        result = boundary.classify_feature(proposal)
        assert result.level == DiscoveryLevel.TIMESCALE
        assert result.capability == CapabilityCategory.DIRECTED

    def test_existing_timescale_is_recombination(self):
        boundary = _make_boundary()
        proposal = FeatureProposal(
            data_streams_required=["close_price"],
            transforms_required=["rsi"],
            timescale="5min",  # In default timescales
        )

        result = boundary.classify_feature(proposal)
        assert result.level == DiscoveryLevel.RECOMBINATION


# ==============================================================================
# Test 3: Classify Novel Computation
# ==============================================================================

class TestClassifyNovelComputation:
    """test_classify_novel_computation: Existing stream + new transform → Level 3."""

    def test_novel_computation(self):
        boundary = _make_boundary()
        proposal = FeatureProposal(
            data_streams_required=["close_price"],
            transforms_required=["wavelet_decomposition"],  # Not in library
        )

        result = boundary.classify_feature(proposal)
        assert result.level == DiscoveryLevel.NOVEL_COMPUTATION
        assert result.capability == CapabilityCategory.DIRECTED

    def test_novel_computation_has_human_actions(self):
        boundary = _make_boundary()
        proposal = FeatureProposal(
            data_streams_required=["close_price"],
            transforms_required=["neural_embedding"],
        )

        result = boundary.classify_feature(proposal)
        assert len(result.human_actions) > 0
        assert any("neural_embedding" in action for action in result.human_actions)


# ==============================================================================
# Test 4: Classify Novel Data
# ==============================================================================

class TestClassifyNovelData:
    """test_classify_novel_data: New data stream → Level 4."""

    def test_novel_data(self):
        boundary = _make_boundary()
        proposal = FeatureProposal(
            data_streams_required=["satellite_imagery"],  # Not in library
            transforms_required=["rolling_mean"],
        )

        result = boundary.classify_feature(proposal)
        assert result.level == DiscoveryLevel.NOVEL_DATA
        assert result.capability == CapabilityCategory.CREATIVE

    def test_novel_data_has_creative_actions(self):
        boundary = _make_boundary()
        proposal = FeatureProposal(
            data_streams_required=["social_sentiment", "news_feed"],
            transforms_required=["rsi"],
        )

        result = boundary.classify_feature(proposal)
        assert len(result.human_actions) > 0
        assert len(result.autonomous_actions) > 0  # Can still characterize gap


# ==============================================================================
# Test 5: Capability Mapping
# ==============================================================================

class TestCapabilityMapping:
    """test_capability_mapping: Each level maps to correct capability."""

    def test_all_levels_mapped(self):
        for level in DiscoveryLevel:
            assert level in LEVEL_CAPABILITY_MAP

    def test_recombination_autonomous(self):
        assert LEVEL_CAPABILITY_MAP[DiscoveryLevel.RECOMBINATION] == CapabilityCategory.AUTONOMOUS

    def test_timescale_autonomous(self):
        assert LEVEL_CAPABILITY_MAP[DiscoveryLevel.TIMESCALE] == CapabilityCategory.AUTONOMOUS

    def test_novel_computation_directed(self):
        assert LEVEL_CAPABILITY_MAP[DiscoveryLevel.NOVEL_COMPUTATION] == CapabilityCategory.DIRECTED

    def test_novel_data_creative(self):
        assert LEVEL_CAPABILITY_MAP[DiscoveryLevel.NOVEL_DATA] == CapabilityCategory.CREATIVE


# ==============================================================================
# Test 6: Boundary Surface Size
# ==============================================================================

class TestBoundarySurfaceSize:
    """test_boundary_surface_size: Computes autonomous space correctly."""

    def test_surface_nonzero(self):
        boundary = _make_boundary()
        surface = boundary.characterize_boundary_surface()

        assert surface.total_autonomous_features > 0
        assert surface.recombination_space > 0
        assert surface.timescale_space > 0
        assert surface.total_transforms == 9
        assert surface.total_data_streams == 4

    def test_coverage_estimate_reasonable(self):
        boundary = _make_boundary()
        surface = boundary.characterize_boundary_surface()

        assert 0.0 < surface.coverage_estimate <= 1.0

    def test_empty_library(self):
        boundary = DiscoveryBoundary(computational_library=ComputationalLibrary())
        surface = boundary.characterize_boundary_surface()
        assert surface.total_autonomous_features == 0


# ==============================================================================
# Test 7: Expand Boundary Transform
# ==============================================================================

class TestExpandBoundaryTransform:
    """test_expand_boundary_transform: New transform expands Level 1 space."""

    def test_transform_expands(self):
        boundary = _make_boundary()
        before = boundary.characterize_boundary_surface()

        expansion = boundary.expand_boundary(
            "transform",
            TransformSpec(name="wavelet", category="signal_processing"),
        )

        after = boundary.characterize_boundary_surface()
        assert after.total_autonomous_features > before.total_autonomous_features
        assert expansion.new_autonomous_features > 0
        assert expansion.expansion_type == "transform"

    def test_transform_creates_level_1(self):
        boundary = _make_boundary()
        expansion = boundary.expand_boundary(
            "transform",
            TransformSpec(name="entropy", category="information_theory"),
        )

        assert len(expansion.new_level_1_features) > 0


# ==============================================================================
# Test 8: Expand Boundary Data
# ==============================================================================

class TestExpandBoundaryData:
    """test_expand_boundary_data: New data stream expands Level 1 space."""

    def test_data_stream_expands(self):
        boundary = _make_boundary()
        before = boundary.characterize_boundary_surface()

        expansion = boundary.expand_boundary(
            "data_stream",
            DataStreamSpec(name="options_flow", category="derivatives"),
        )

        after = boundary.characterize_boundary_surface()
        assert after.total_autonomous_features > before.total_autonomous_features
        assert expansion.new_autonomous_features > 0
        assert expansion.expansion_type == "data_stream"


# ==============================================================================
# Test 9: Research Brief Contains Signature
# ==============================================================================

class TestResearchBriefContainsSignature:
    """test_research_brief_contains_signature: Brief includes gap description."""

    def test_brief_has_gap_info(self):
        boundary = _make_boundary()

        # Create a mock signature
        class MockSignature:
            n_misses = 30
            temporal_profile = type('TP', (), {'peak_hours': [9, 10], 'has_strong_pattern': True})()
            regime_profile = type('RP', (), {'regime_counts': {'BULL': 10, 'BEAR': 20}})()
            lead_lag_profile = type('LL', (), {'leading_instruments': ['SPY']})()
            def summary(self):
                return "30 missed trades concentrated in morning hours"

        brief = boundary.generate_human_research_brief(
            signature=MockSignature(),
        )

        assert brief.gap_description != ""
        assert "30" in brief.severity
        assert brief.priority == "MEDIUM"  # 30 > 20

    def test_high_severity_brief(self):
        boundary = _make_boundary()

        class MockSignature:
            n_misses = 100
            temporal_profile = None
            regime_profile = None
            lead_lag_profile = None
            def summary(self):
                return "100 missed trades"

        brief = boundary.generate_human_research_brief(signature=MockSignature())
        assert brief.priority == "HIGH"


# ==============================================================================
# Test 10: Research Brief Contains Suggestions
# ==============================================================================

class TestResearchBriefContainsSuggestions:
    """test_research_brief_contains_suggestions: Brief includes search directions."""

    def test_brief_has_search_directions(self):
        boundary = _make_boundary()

        class MockSignature:
            n_misses = 25
            temporal_profile = type('TP', (), {
                'peak_hours': [9, 10],
                'has_strong_pattern': True,
            })()
            regime_profile = None
            lead_lag_profile = type('LL', (), {
                'leading_instruments': ['VIX'],
            })()
            def summary(self):
                return "Gap detected"

        brief = boundary.generate_human_research_brief(signature=MockSignature())

        assert len(brief.search_directions) > 0
        # Should have both temporal and cross-asset suggestions
        assert any("event-driven" in d.lower() or "temporal" in d.lower()
                    for d in brief.search_directions)
        assert any("VIX" in d for d in brief.search_directions)

    def test_brief_default_direction(self):
        """Brief without strong signals should have generic direction."""
        boundary = _make_boundary()
        brief = boundary.generate_human_research_brief()

        assert len(brief.search_directions) > 0


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestEdgeCases:

    def test_empty_proposal(self):
        boundary = _make_boundary()
        proposal = FeatureProposal()
        result = boundary.classify_feature(proposal)
        # No streams, no transforms → recombination (trivial)
        assert result.level == DiscoveryLevel.RECOMBINATION

    def test_discovery_level_values(self):
        assert DiscoveryLevel.RECOMBINATION.value == 1
        assert DiscoveryLevel.NOVEL_DATA.value == 4

    def test_capability_values(self):
        assert CapabilityCategory.AUTONOMOUS.value == "autonomous"
        assert CapabilityCategory.CREATIVE.value == "creative_human"

    def test_invalid_expansion(self):
        boundary = _make_boundary()
        expansion = boundary.expand_boundary("invalid", None)
        assert expansion.new_autonomous_features == 0
