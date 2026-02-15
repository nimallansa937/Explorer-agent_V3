"""
Tests for Feature Registry & Projection Layer - Phase 2

Explorer Prime v2.0
20+ tests covering: registry, schema management, projector,
maturity pipeline, and shadow re-evolution.
"""

import numpy as np
import pytest
from datetime import datetime, timedelta

from shared.feature_registry import (
    FeatureStatus,
    DataClass,
    MaturityDecision,
    FeatureDefinition,
    TrialRecord,
    FeatureRegistry,
    FeatureProjector,
    FeatureMaturityPipeline,
    ShadowReEvolution,
)


# ==============================================================================
# Test 1: Registry pre-populated with 60 CORE features
# ==============================================================================

class TestRegistryPrePopulated:
    """Verify 60 CORE features on init."""

    def test_registry_pre_populated_60_core(self):
        """Registry initializes with exactly 60 CORE features."""
        registry = FeatureRegistry()
        core = registry.get_core_schema()
        assert len(core) == 60

    def test_all_pre_populated_are_core(self):
        """All pre-populated features have CORE status."""
        registry = FeatureRegistry()
        for fid in registry.get_core_schema():
            feat = registry.get_feature(fid)
            assert feat is not None
            assert feat.status == FeatureStatus.CORE

    def test_data_class_distribution(self):
        """Features distributed across correct data classes."""
        registry = FeatureRegistry()
        assert len(registry.get_features_by_class(DataClass.PRICE)) == 20
        assert len(registry.get_features_by_class(DataClass.VOLUME)) == 10
        assert len(registry.get_features_by_class(DataClass.ORDER_FLOW)) == 10
        assert len(registry.get_features_by_class(DataClass.VOLATILITY)) == 8
        assert len(registry.get_features_by_class(DataClass.MICROSTRUCTURE)) == 6
        assert len(registry.get_features_by_class(DataClass.REGIME)) == 6


# ==============================================================================
# Test 2: Active schema includes validated
# ==============================================================================

class TestActiveSchema:
    """CORE + VALIDATED features in active schema."""

    def test_active_schema_includes_validated(self):
        """Active schema includes both CORE and VALIDATED features."""
        registry = FeatureRegistry()

        # Add a VALIDATED feature
        registry.register("test_validated", "Test Validated",
                          DataClass.DERIVED, status=FeatureStatus.VALIDATED)

        active = registry.get_active_schema()
        assert "test_validated" in active
        assert len(active) == 61  # 60 CORE + 1 VALIDATED

    def test_active_schema_excludes_experimental(self):
        """EXPERIMENTAL features not in active schema."""
        registry = FeatureRegistry()
        registry.register("test_exp", "Test Experimental",
                          DataClass.DERIVED, status=FeatureStatus.EXPERIMENTAL)

        active = registry.get_active_schema()
        assert "test_exp" not in active

    def test_active_schema_excludes_deprecated(self):
        """DEPRECATED features not in active schema."""
        registry = FeatureRegistry()
        registry.register("test_dep", "Test Deprecated",
                          DataClass.DERIVED, status=FeatureStatus.VALIDATED)
        registry.deprecate("test_dep")

        active = registry.get_active_schema()
        assert "test_dep" not in active


# ==============================================================================
# Test 3: Core schema excludes validated
# ==============================================================================

class TestCoreSchema:
    """CORE only in core schema."""

    def test_core_schema_excludes_validated(self):
        """Core schema only includes CORE features."""
        registry = FeatureRegistry()
        registry.register("test_val", "Test Validated",
                          DataClass.DERIVED, status=FeatureStatus.VALIDATED)

        core = registry.get_core_schema()
        assert "test_val" not in core
        assert len(core) == 60


# ==============================================================================
# Test 4: Promote increments version
# ==============================================================================

class TestPromoteIncrementsVersion:
    """Status change to/from CORE bumps schema version."""

    def test_promote_increments_version(self):
        """Promoting to CORE increments schema version."""
        registry = FeatureRegistry()
        initial_version = registry.schema_version

        registry.register("new_feat", "New Feature",
                          DataClass.DERIVED, status=FeatureStatus.EXPERIMENTAL)
        # EXPERIMENTAL → VALIDATED (no version bump)
        registry.promote("new_feat", FeatureStatus.VALIDATED)
        assert registry.schema_version == initial_version

        # VALIDATED → CORE (version bump!)
        registry.promote("new_feat", FeatureStatus.CORE)
        assert registry.schema_version != initial_version

    def test_invalid_promotion_raises(self):
        """Invalid status transitions raise ValueError."""
        registry = FeatureRegistry()
        registry.register("feat", "Feature",
                          DataClass.DERIVED, status=FeatureStatus.EXPERIMENTAL)

        with pytest.raises(ValueError, match="Invalid transition"):
            registry.promote("feat", FeatureStatus.CORE)  # Can't skip VALIDATED


# ==============================================================================
# Test 5: Deprecate records history
# ==============================================================================

class TestDeprecateRecordsHistory:
    """Deprecated feature tracked in version history."""

    def test_deprecate_records_history(self):
        """Deprecation from CORE is recorded in version history."""
        registry = FeatureRegistry()
        initial_history_len = len(registry.version_history)

        # Deprecate a CORE feature
        core_features = registry.get_core_schema()
        fid = core_features[0]
        registry.deprecate(fid)

        assert len(registry.version_history) > initial_history_len
        latest = registry.version_history[-1]
        assert fid in latest[2]  # Description mentions feature

    def test_deprecate_sets_deprecated_version(self):
        """Deprecated feature records which version it was deprecated in."""
        registry = FeatureRegistry()
        registry.register("temp_feat", "Temp", DataClass.DERIVED,
                          status=FeatureStatus.VALIDATED)
        registry.deprecate("temp_feat")

        feat = registry.get_feature("temp_feat")
        assert feat.deprecated_version is not None


# ==============================================================================
# Test 6: Projector extract
# ==============================================================================

class TestProjectorExtract:
    """Correct subset extracted from full vector."""

    def test_projector_extract(self):
        """Project extracts correct feature subset."""
        registry = FeatureRegistry()
        projector = FeatureProjector(registry)

        active = registry.get_active_schema()
        full_vec = np.arange(len(active), dtype=np.float64)

        # Extract first 10 features
        target = active[:10]
        projected = projector.project(full_vec, target)

        assert len(projected) == 10
        for i in range(10):
            assert projected[i] == pytest.approx(float(i))

    def test_projector_pad_to_full(self):
        """Pad expands partial vector correctly."""
        registry = FeatureRegistry()
        projector = FeatureProjector(registry)

        active = registry.get_active_schema()
        partial = np.array([1.0, 2.0, 3.0])
        source_schema = active[:3]

        padded = projector.pad_to_full(partial, source_schema)
        assert len(padded) == len(active)
        assert padded[0] == 1.0
        assert padded[1] == 2.0
        assert padded[2] == 3.0
        assert padded[3] == 0.0  # Zero-padded


# ==============================================================================
# Test 7: Schema-aware distance
# ==============================================================================

class TestSchemaAwareDistance:
    """Shared-dimension distance works correctly."""

    def test_projector_schema_aware_distance(self):
        """Distance computation over shared dimensions."""
        registry = FeatureRegistry()
        projector = FeatureProjector(registry)

        core = registry.get_core_schema()
        vec_a = np.ones(len(core))
        vec_b = np.ones(len(core))

        # Identical vectors should have ~0 distance
        dist = projector.schema_aware_distance(vec_a, vec_b, core, core)
        assert dist == pytest.approx(0.0, abs=0.01)

    def test_projector_orthogonal_vectors(self):
        """Orthogonal vectors should have positive distance."""
        registry = FeatureRegistry()
        projector = FeatureProjector(registry)

        core = registry.get_core_schema()
        vec_a = np.zeros(len(core))
        vec_b = np.zeros(len(core))
        vec_a[0] = 1.0
        vec_b[1] = 1.0

        dist = projector.schema_aware_distance(vec_a, vec_b, core, core)
        assert dist > 0.0


# ==============================================================================
# Test 8: Minimum shared features
# ==============================================================================

class TestMinSharedFeatures:
    """Returns 1.0 if <5 shared features."""

    def test_projector_min_shared_features(self):
        """Distance returns 1.0 when fewer than 5 shared features."""
        registry = FeatureRegistry()
        projector = FeatureProjector(registry)

        # Two schemas with only 3 shared features
        schema_a = ["feat_1", "feat_2", "feat_3", "feat_4", "feat_5"]
        schema_b = ["feat_1", "feat_2", "feat_3", "feat_x", "feat_y"]

        vec_a = np.ones(5)
        vec_b = np.ones(5)

        dist = projector.schema_aware_distance(vec_a, vec_b, schema_a, schema_b)
        # 3 shared < 5 minimum → returns 1.0
        assert dist == 1.0


# ==============================================================================
# Test 9: Coverage normalization
# ==============================================================================

class TestCoverageNormalization:
    """Jaccard weighting correct."""

    def test_projector_coverage_normalization(self):
        """Distance is divided by Jaccard coverage."""
        registry = FeatureRegistry()
        projector = FeatureProjector(registry)

        core = registry.get_core_schema()

        # Same schema, all features shared
        vec_a = np.random.randn(len(core))
        vec_b = np.random.randn(len(core))

        dist_full = projector.schema_aware_distance(vec_a, vec_b, core, core)

        # Partial overlap: add 10 unique features to each schema
        schema_a = core + [f"extra_a_{i}" for i in range(10)]
        schema_b = core + [f"extra_b_{i}" for i in range(10)]
        vec_a_ext = np.concatenate([vec_a, np.random.randn(10)])
        vec_b_ext = np.concatenate([vec_b, np.random.randn(10)])

        dist_partial = projector.schema_aware_distance(vec_a_ext, vec_b_ext,
                                                       schema_a, schema_b)

        # Partial coverage should amplify distance (divided by coverage < 1.0)
        # Coverage = 60 / 80 = 0.75, so dist_partial ≈ dist_full / 0.75
        assert dist_partial >= dist_full - 0.01  # Partial should be >= full


# ==============================================================================
# Test 10: Maturity promote all pass
# ==============================================================================

class TestMaturityPromote:
    """All criteria met → PROMOTE."""

    def test_maturity_promote_all_pass(self):
        """Feature with all criteria met gets promoted."""
        registry = FeatureRegistry()
        registry.register("good_feat", "Good Feature", DataClass.DERIVED,
                          status=FeatureStatus.EXPERIMENTAL)

        pipeline = FeatureMaturityPipeline(registry)
        pipeline.start_trial("good_feat")
        pipeline.update_trial("good_feat",
                              strategies_generated=600,
                              hifa_passed=25,
                              shadow_entries=5,
                              shadow_passes=2)

        decision = pipeline.evaluate_trial(
            "good_feat",
            functional_correlation=0.3,
            base_pass_rate=0.03,
        )
        assert decision == MaturityDecision.PROMOTE


# ==============================================================================
# Test 11: Deprecate low samples
# ==============================================================================

class TestMaturityDeprecateLowSamples:
    """<500 strategies → EXTEND then DEPRECATE."""

    def test_maturity_deprecate_low_samples(self):
        """Too few strategies: first EXTEND, then DEPRECATE."""
        registry = FeatureRegistry()
        registry.register("low_feat", "Low Feature", DataClass.DERIVED,
                          status=FeatureStatus.EXPERIMENTAL)

        pipeline = FeatureMaturityPipeline(registry)
        pipeline.start_trial("low_feat")
        pipeline.update_trial("low_feat", strategies_generated=100,
                              hifa_passed=5, shadow_entries=1, shadow_passes=0)

        # First evaluation: EXTEND
        decision1 = pipeline.evaluate_trial("low_feat", functional_correlation=0.3)
        assert decision1 == MaturityDecision.EXTEND

        # Second evaluation (still low): DEPRECATE
        decision2 = pipeline.evaluate_trial("low_feat", functional_correlation=0.3)
        assert decision2 == MaturityDecision.DEPRECATE


# ==============================================================================
# Test 12: Deprecate redundant
# ==============================================================================

class TestMaturityDeprecateRedundant:
    """Correlation >0.7 → DEPRECATE."""

    def test_maturity_deprecate_redundant(self):
        """High functional correlation with core → DEPRECATE."""
        registry = FeatureRegistry()
        registry.register("redundant", "Redundant Feature", DataClass.DERIVED,
                          status=FeatureStatus.EXPERIMENTAL)

        pipeline = FeatureMaturityPipeline(registry)
        pipeline.start_trial("redundant")
        pipeline.update_trial("redundant",
                              strategies_generated=600,
                              hifa_passed=20,
                              shadow_entries=4,
                              shadow_passes=2)

        decision = pipeline.evaluate_trial(
            "redundant",
            functional_correlation=0.85,  # > 0.7 threshold
            base_pass_rate=0.03,
        )
        assert decision == MaturityDecision.DEPRECATE


# ==============================================================================
# Test 13: Extend once only
# ==============================================================================

class TestMaturityExtendOnce:
    """Second extension → DEPRECATE."""

    def test_maturity_extend_once_only(self):
        """Can only extend once, then must deprecate."""
        registry = FeatureRegistry()
        registry.register("slow_feat", "Slow Feature", DataClass.DERIVED,
                          status=FeatureStatus.EXPERIMENTAL)

        pipeline = FeatureMaturityPipeline(registry)
        pipeline.start_trial("slow_feat")
        pipeline.update_trial("slow_feat", strategies_generated=200)

        # First: EXTEND
        d1 = pipeline.evaluate_trial("slow_feat")
        assert d1 == MaturityDecision.EXTEND

        # Second: DEPRECATE (already extended)
        d2 = pipeline.evaluate_trial("slow_feat")
        assert d2 == MaturityDecision.DEPRECATE


# ==============================================================================
# Test 14: Functional correlation uses performance
# ==============================================================================

class TestFunctionalCorrelation:
    """Not raw feature correlation."""

    def test_functional_correlation_uses_performance(self):
        """Functional correlation compares strategy performance, not raw features."""
        registry = FeatureRegistry()
        pipeline = FeatureMaturityPipeline(registry)

        # Create performance vectors (Sharpe contributions)
        feat_perf = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # Highly correlated core feature
        core_vectors = {
            "core_1": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            "core_2": np.random.randn(10),
        }

        corr = pipeline.compute_functional_correlation(feat_perf, core_vectors)

        # Should be high (close to 1.0) because feat_perf matches core_1
        assert corr > 0.9

    def test_functional_correlation_uncorrelated(self):
        """Uncorrelated performance vectors produce low correlation."""
        registry = FeatureRegistry()
        pipeline = FeatureMaturityPipeline(registry)

        np.random.seed(42)
        feat_perf = np.random.randn(50)
        core_vectors = {
            "core_1": np.random.randn(50),
            "core_2": np.random.randn(50),
        }

        corr = pipeline.compute_functional_correlation(feat_perf, core_vectors)
        assert corr < 0.5  # Should be low


# ==============================================================================
# Test 15: Shadow re-evolution spawns search
# ==============================================================================

class TestShadowReEvolution:
    """Legacy strategies trigger re-evolution."""

    def test_shadow_reevolution_spawns_search(self):
        """Schema change triggers re-evolution tasks for old-schema strategies."""
        registry = FeatureRegistry()
        reevo = ShadowReEvolution(registry)

        # Simulate schema change
        registry.register("new_core", "New Core", DataClass.DERIVED,
                          status=FeatureStatus.EXPERIMENTAL)
        registry.promote("new_core", FeatureStatus.VALIDATED)
        registry.promote("new_core", FeatureStatus.CORE)

        # Production strategies on old schema
        prod_strats = [
            {"strategy_id": "s1", "schema_version": "1.0.0", "performance_sharpe": 1.5},
            {"strategy_id": "s2", "schema_version": "1.0.0", "performance_sharpe": 0.8},
            {"strategy_id": "s3", "schema_version": registry.schema_version, "performance_sharpe": 1.2},
        ]

        tasks = reevo.check_for_reevolution(
            prod_strats, registry.get_active_schema()
        )

        # s1 and s2 need re-evolution, s3 is current
        assert len(tasks) == 2
        assert all(t["old_version"] == "1.0.0" for t in tasks)
        assert all(t["new_version"] == registry.schema_version for t in tasks)


# ==============================================================================
# Test 16: Legacy risk cap
# ==============================================================================

class TestLegacyRiskCap:
    """LEGACY strategies capped at 5%."""

    def test_legacy_risk_cap(self):
        """Legacy risk allocation checked against 5% cap."""
        registry = FeatureRegistry()
        reevo = ShadowReEvolution(registry)

        # Tag some strategies as legacy
        reevo.tag_as_legacy("s1")
        reevo.tag_as_legacy("s2")

        portfolio = [
            {"strategy_id": "s1", "risk_weight": 0.02},
            {"strategy_id": "s2", "risk_weight": 0.02},
            {"strategy_id": "s3", "risk_weight": 0.96},
        ]

        # 0.02 + 0.02 = 0.04 < 0.05 → within cap
        assert reevo.is_legacy_risk_within_cap(portfolio) is True

        # Increase legacy risk
        portfolio[0]["risk_weight"] = 0.04
        # 0.04 + 0.02 = 0.06 > 0.05 → exceeds cap
        assert reevo.is_legacy_risk_within_cap(portfolio) is False


# ==============================================================================
# Test 17: Schema version history
# ==============================================================================

class TestSchemaVersionHistory:
    """Full history retrievable."""

    def test_schema_version_history(self):
        """Version history tracks all changes."""
        registry = FeatureRegistry()
        initial_len = len(registry.version_history)

        # Make some changes
        registry.register("new1", "New 1", DataClass.DERIVED,
                          status=FeatureStatus.EXPERIMENTAL)
        registry.promote("new1", FeatureStatus.VALIDATED)
        registry.promote("new1", FeatureStatus.CORE)

        history = registry.version_history
        assert len(history) > initial_len

        # Each entry is (version, datetime, description)
        for version, ts, desc in history:
            assert isinstance(version, str)
            assert isinstance(ts, datetime)
            assert isinstance(desc, str)


# ==============================================================================
# Test 18: Feature by class
# ==============================================================================

class TestFeatureByClass:
    """Correct features returned per DataClass."""

    def test_feature_by_class(self):
        """get_features_by_class returns correct subset."""
        registry = FeatureRegistry()

        price = registry.get_features_by_class(DataClass.PRICE)
        assert len(price) == 20
        assert all(f.data_class == DataClass.PRICE for f in price)

        vol = registry.get_features_by_class(DataClass.VOLATILITY)
        assert len(vol) == 8
        assert all(f.data_class == DataClass.VOLATILITY for f in vol)

    def test_empty_class(self):
        """Empty data class returns empty list."""
        registry = FeatureRegistry()
        cross = registry.get_features_by_class(DataClass.CROSS_ASSET)
        assert len(cross) == 0  # No pre-populated CROSS_ASSET features


# ==============================================================================
# Test 19: Backward compatible 60-dim
# ==============================================================================

class TestBackwardCompat60dim:
    """60-dim projections match v1.0 behavior."""

    def test_backward_compatible_60dim(self):
        """Core schema produces 60-dim vector matching v1.0."""
        registry = FeatureRegistry()
        projector = FeatureProjector(registry)

        core = registry.get_core_schema()
        assert len(core) == 60

        # Full vector aligned to core schema
        full_vec = np.arange(60, dtype=np.float64)

        # Project to core schema should be identity
        projected = projector.project(full_vec, core)
        np.testing.assert_array_almost_equal(projected, full_vec)


# ==============================================================================
# Test 20: Cross-version distance
# ==============================================================================

class TestCrossVersionDistance:
    """v1.0 vs v2.0 strategies compared correctly."""

    def test_cross_version_distance(self):
        """Distance between v1.0 and v2.0 schemas works correctly."""
        registry = FeatureRegistry()
        projector = FeatureProjector(registry)

        core_v1 = registry.get_core_schema()

        # Simulate v2.0: add some features
        registry.register("new_feat_1", "New 1", DataClass.DERIVED,
                          status=FeatureStatus.VALIDATED)
        registry.register("new_feat_2", "New 2", DataClass.DERIVED,
                          status=FeatureStatus.VALIDATED)

        active_v2 = registry.get_active_schema()  # 60 CORE + 2 VALIDATED

        # Create vectors
        vec_v1 = np.random.randn(len(core_v1))
        vec_v2 = np.random.randn(len(active_v2))

        # Distance should be computable (60 shared features >= 5 minimum)
        dist = projector.schema_aware_distance(vec_v1, vec_v2, core_v1, active_v2)
        assert 0.0 <= dist  # Valid distance
        assert dist < 10.0  # Reasonable range


# ==============================================================================
# Additional edge case tests
# ==============================================================================

class TestRegistryEdgeCases:
    """Edge cases and error handling."""

    def test_duplicate_registration_raises(self):
        """Registering same feature ID twice raises error."""
        registry = FeatureRegistry()
        with pytest.raises(ValueError, match="already registered"):
            registry.register("price_rsi_14", "Duplicate RSI", DataClass.PRICE)

    def test_promote_unknown_feature_raises(self):
        """Promoting unknown feature raises KeyError."""
        registry = FeatureRegistry()
        with pytest.raises(KeyError):
            registry.promote("nonexistent", FeatureStatus.VALIDATED)

    def test_deprecate_idempotent(self):
        """Deprecating already deprecated feature is no-op."""
        registry = FeatureRegistry()
        registry.register("temp", "Temp", DataClass.DERIVED,
                          status=FeatureStatus.VALIDATED)
        registry.deprecate("temp")
        registry.deprecate("temp")  # Should not raise

    def test_empty_registry(self):
        """Registry without pre-population is empty."""
        registry = FeatureRegistry(pre_populate=False)
        assert registry.core_count == 0
        assert registry.feature_count == 0
        assert len(registry.get_core_schema()) == 0

    def test_schema_snapshot_on_version_change(self):
        """Schema snapshot is saved when version changes."""
        registry = FeatureRegistry()
        initial_version = registry.schema_version

        registry.register("new", "New", DataClass.DERIVED,
                          status=FeatureStatus.EXPERIMENTAL)
        registry.promote("new", FeatureStatus.VALIDATED)
        registry.promote("new", FeatureStatus.CORE)

        new_version = registry.schema_version
        assert new_version != initial_version

        # Both versions should have snapshots
        snap_old = registry.get_schema_at_version(initial_version)
        snap_new = registry.get_schema_at_version(new_version)

        assert len(snap_old) == 60
        assert len(snap_new) == 61  # 60 + 1 new CORE


class TestProjectorEdgeCases:
    """Projector edge cases."""

    def test_zero_vectors(self):
        """Zero vectors produce max distance."""
        registry = FeatureRegistry()
        projector = FeatureProjector(registry)
        core = registry.get_core_schema()

        dist = projector.schema_aware_distance(
            np.zeros(60), np.zeros(60), core, core
        )
        assert dist == 1.0  # Zero norm → max distance

    def test_distance_symmetry(self):
        """Distance is symmetric: d(a,b) == d(b,a)."""
        registry = FeatureRegistry()
        projector = FeatureProjector(registry)
        core = registry.get_core_schema()

        np.random.seed(123)
        vec_a = np.random.randn(60)
        vec_b = np.random.randn(60)

        d_ab = projector.schema_aware_distance(vec_a, vec_b, core, core)
        d_ba = projector.schema_aware_distance(vec_b, vec_a, core, core)
        assert d_ab == pytest.approx(d_ba, abs=1e-10)
