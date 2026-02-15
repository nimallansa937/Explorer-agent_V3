"""
Discovery Boundary Formalization — Four-Level Discovery Taxonomy

Formalizes the boundary between autonomous exploration (C2) and
human domain expertise (C1). The Discovery Boundary is the formal
encoding of the balloon surface:

- Level 1 (Recombination): Inside the balloon → AUTONOMOUS
- Level 2 (Timescale): Inside if configured → AUTONOMOUS/DIRECTED
- Level 3 (Novel Computation): On the surface → DIRECTED
- Level 4 (Novel Data): Outside → CREATIVE

Explorer Prime v2.0 - Phase 8
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
from datetime import datetime


# ==============================================================================
# Enums
# ==============================================================================

class DiscoveryLevel(Enum):
    """Four-level discovery taxonomy."""
    RECOMBINATION = 1       # New combinations of existing features
    TIMESCALE = 2           # Existing streams at different temporal resolution
    NOVEL_COMPUTATION = 3   # Existing data, new transformation
    NOVEL_DATA = 4          # Entirely new data modality


class CapabilityCategory(Enum):
    """Who can implement the discovery."""
    AUTONOMOUS = "autonomous"           # Pipeline finds AND implements
    DIRECTED = "directed_human"         # Pipeline identifies gap shape, human implements
    CREATIVE = "creative_human"         # Human imagines, pipeline validates


# Default mapping from level to capability
LEVEL_CAPABILITY_MAP: Dict[DiscoveryLevel, CapabilityCategory] = {
    DiscoveryLevel.RECOMBINATION: CapabilityCategory.AUTONOMOUS,
    DiscoveryLevel.TIMESCALE: CapabilityCategory.AUTONOMOUS,  # IF timescale_sweep configured
    DiscoveryLevel.NOVEL_COMPUTATION: CapabilityCategory.DIRECTED,
    DiscoveryLevel.NOVEL_DATA: CapabilityCategory.CREATIVE,
}


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class TransformSpec:
    """Specification of a computational transform."""
    name: str
    category: str  # e.g., "rolling_window_stats", "momentum_indicators"
    input_type: str = "timeseries"
    output_type: str = "feature"
    description: str = ""


@dataclass
class DataStreamSpec:
    """Specification of a data stream."""
    name: str
    category: str  # e.g., "price", "volume", "order_flow", "alternative"
    timescales: List[str] = field(default_factory=lambda: ["1min", "5min", "15min"])
    description: str = ""


@dataclass
class DiscoveryClassification:
    """Result of classifying a proposed feature's discovery level."""
    level: DiscoveryLevel
    capability: CapabilityCategory
    rationale: str = ""
    autonomous_actions: List[str] = field(default_factory=list)
    human_actions: List[str] = field(default_factory=list)
    gap_signature: Any = None  # AnomalySignature


@dataclass
class BoundarySurface:
    """Description of the current autonomous exploration boundary."""
    total_autonomous_features: int = 0
    recombination_space: int = 0         # Level 1 feature count
    timescale_space: int = 0             # Level 2 feature count
    total_transforms: int = 0
    total_data_streams: int = 0
    coverage_estimate: float = 0.0       # autonomous / estimated total
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryExpansion:
    """Result of expanding the discovery boundary."""
    expansion_type: str = ""             # "transform" or "data_stream"
    addition_name: str = ""
    new_autonomous_features: int = 0
    description: str = ""
    new_level_1_features: List[str] = field(default_factory=list)


@dataclass
class ResearchBrief:
    """Structured document for human researchers."""
    brief_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    gap_description: str = ""
    temporal_profile: str = ""
    regime_context: str = ""
    severity: str = ""
    features_tried: List[str] = field(default_factory=list)
    structures_tested: List[str] = field(default_factory=list)
    missing_signal_type: str = ""
    search_directions: List[str] = field(default_factory=list)
    priority: str = "MEDIUM"
    estimated_alpha_opportunity: float = 0.0
    generated_at: datetime = field(default_factory=datetime.utcnow)


# ==============================================================================
# Computational Library
# ==============================================================================

class ComputationalLibrary:
    """Tracks available computational transforms and data streams.

    Standard categories:
    - rolling_window_stats: mean, std, skew, kurtosis
    - momentum_indicators: RSI, MACD, Bollinger, etc.
    - order_flow_metrics: OFI, imbalance, aggression
    - volatility_estimators: realized, Garman-Klass, Yang-Zhang
    - regime_detectors: HMM, threshold-based
    """

    def __init__(self):
        self._transforms: Dict[str, TransformSpec] = {}
        self._data_streams: Dict[str, DataStreamSpec] = {}
        self._configured_timescales: List[str] = ["1min", "5min", "15min", "1hour", "1day"]
        self._timescale_sweep_configured: bool = True

    def add_transform(self, spec: TransformSpec) -> None:
        """Register a new computational transform."""
        self._transforms[spec.name] = spec

    def add_data_stream(self, spec: DataStreamSpec) -> None:
        """Register a new data stream."""
        self._data_streams[spec.name] = spec

    def has_transform(self, name: str) -> bool:
        """Check if a transform exists in the library."""
        return name in self._transforms

    def has_data_stream(self, name: str) -> bool:
        """Check if a data stream exists."""
        return name in self._data_streams

    def get_transforms(self) -> Dict[str, TransformSpec]:
        return dict(self._transforms)

    def get_data_streams(self) -> Dict[str, DataStreamSpec]:
        return dict(self._data_streams)

    @property
    def timescale_sweep_configured(self) -> bool:
        return self._timescale_sweep_configured

    @timescale_sweep_configured.setter
    def timescale_sweep_configured(self, value: bool) -> None:
        self._timescale_sweep_configured = value

    @property
    def configured_timescales(self) -> List[str]:
        return list(self._configured_timescales)


# ==============================================================================
# Feature Proposal (for classification)
# ==============================================================================

@dataclass
class FeatureProposal:
    """A proposed feature to be classified by the Discovery Boundary."""
    feature_id: str = ""
    name: str = ""
    data_streams_required: List[str] = field(default_factory=list)
    transforms_required: List[str] = field(default_factory=list)
    timescale: Optional[str] = None
    description: str = ""


# ==============================================================================
# Discovery Boundary
# ==============================================================================

class DiscoveryBoundary:
    """Classifies features by discovery level and manages the boundary surface.

    The four-level taxonomy:
    1. RECOMBINATION: Existing streams + existing transforms → AUTONOMOUS
    2. TIMESCALE: Existing streams at different resolution → AUTONOMOUS/DIRECTED
    3. NOVEL_COMPUTATION: Existing streams + new transforms → DIRECTED
    4. NOVEL_DATA: New data modality → CREATIVE
    """

    # Estimated total useful feature space (from literature/domain knowledge)
    ESTIMATED_TOTAL_FEATURE_SPACE: int = 10000

    def __init__(
        self,
        feature_registry: Any = None,
        computational_library: Optional[ComputationalLibrary] = None,
    ):
        self.feature_registry = feature_registry
        self.library = computational_library or ComputationalLibrary()

    def classify_feature(
        self,
        proposed_feature: FeatureProposal,
    ) -> DiscoveryClassification:
        """Classify a proposed feature by its discovery level.

        Decision tree:
        1. Uses new data streams? → Level 4 (NOVEL_DATA) → CREATIVE
        2. Uses existing streams + existing transforms? → Level 1 (RECOMBINATION) → AUTONOMOUS
        3. Uses existing streams at different timescale? → Level 2 → AUTONOMOUS/DIRECTED
        4. Uses existing streams + new transforms? → Level 3 (NOVEL_COMPUTATION) → DIRECTED

        Args:
            proposed_feature: Feature to classify

        Returns:
            DiscoveryClassification with level, capability, and actions
        """
        # Check data stream availability
        streams_available = all(
            self.library.has_data_stream(ds)
            for ds in proposed_feature.data_streams_required
        ) if proposed_feature.data_streams_required else True

        # Check transform availability
        transforms_available = all(
            self.library.has_transform(t)
            for t in proposed_feature.transforms_required
        ) if proposed_feature.transforms_required else True

        # Decision logic
        if not streams_available:
            # Novel data required
            missing_streams = [
                ds for ds in proposed_feature.data_streams_required
                if not self.library.has_data_stream(ds)
            ]
            return DiscoveryClassification(
                level=DiscoveryLevel.NOVEL_DATA,
                capability=CapabilityCategory.CREATIVE,
                rationale=f"Requires new data stream(s): {missing_streams}",
                autonomous_actions=[
                    "Identify temporal correlations with missing signal",
                    "Characterize gap shape for human researchers",
                ],
                human_actions=[
                    f"Source and integrate data stream(s): {missing_streams}",
                    "Define preprocessing pipeline for new data",
                    "Validate signal quality and relevance",
                ],
            )

        if streams_available and transforms_available:
            # All components exist — check if it's timescale or recombination
            if proposed_feature.timescale:
                # Has timescale component
                if proposed_feature.timescale in self.library.configured_timescales:
                    # Already configured timescale → pure recombination
                    return DiscoveryClassification(
                        level=DiscoveryLevel.RECOMBINATION,
                        capability=CapabilityCategory.AUTONOMOUS,
                        rationale="All streams, transforms, and timescale already available",
                        autonomous_actions=[
                            "Generate feature automatically",
                            "Run through HIFA validation",
                            "Enter Feature Maturity Pipeline",
                        ],
                        human_actions=[],
                    )
                else:
                    # New timescale
                    if self.library.timescale_sweep_configured:
                        return DiscoveryClassification(
                            level=DiscoveryLevel.TIMESCALE,
                            capability=CapabilityCategory.AUTONOMOUS,
                            rationale=f"Timescale sweep configured, can explore {proposed_feature.timescale}",
                            autonomous_actions=[
                                f"Compute feature at timescale {proposed_feature.timescale}",
                                "Run through HIFA validation",
                            ],
                            human_actions=[],
                        )
                    else:
                        return DiscoveryClassification(
                            level=DiscoveryLevel.TIMESCALE,
                            capability=CapabilityCategory.DIRECTED,
                            rationale=f"Timescale {proposed_feature.timescale} not configured",
                            autonomous_actions=[
                                "Identify that different timescale is needed",
                            ],
                            human_actions=[
                                f"Configure timescale sweep to include {proposed_feature.timescale}",
                            ],
                        )

            # Pure recombination
            return DiscoveryClassification(
                level=DiscoveryLevel.RECOMBINATION,
                capability=CapabilityCategory.AUTONOMOUS,
                rationale="All streams and transforms already available",
                autonomous_actions=[
                    "Generate feature automatically",
                    "Run through HIFA validation",
                    "Enter Feature Maturity Pipeline",
                ],
                human_actions=[],
            )

        # Streams available but transforms not
        missing_transforms = [
            t for t in proposed_feature.transforms_required
            if not self.library.has_transform(t)
        ]
        return DiscoveryClassification(
            level=DiscoveryLevel.NOVEL_COMPUTATION,
            capability=CapabilityCategory.DIRECTED,
            rationale=f"Requires new transform(s): {missing_transforms}",
            autonomous_actions=[
                "Characterize what kind of computation is missing",
                "Identify temporal patterns that suggest signal type",
            ],
            human_actions=[
                f"Implement transform(s): {missing_transforms}",
                "Add to computational library",
                "Validate transform output properties",
            ],
        )

    def characterize_boundary_surface(self) -> BoundarySurface:
        """Enumerate the current autonomous exploration space.

        Computes:
        a. All pairwise combinations of features (Level 1)
        b. All features at all configured timescales (Level 2)
        c. All transforms applied to all data streams

        Returns:
            BoundarySurface with size and coverage estimates
        """
        transforms = self.library.get_transforms()
        streams = self.library.get_data_streams()
        timescales = self.library.configured_timescales

        n_transforms = len(transforms)
        n_streams = len(streams)
        n_timescales = len(timescales)

        # Level 1: each transform applied to each stream
        base_features = n_transforms * n_streams

        # Level 1: pairwise combinations of base features
        if base_features > 1:
            recombination_space = base_features + (base_features * (base_features - 1)) // 2
        else:
            recombination_space = base_features

        # Level 2: each base feature at each timescale
        timescale_space = base_features * n_timescales

        total_autonomous = recombination_space + timescale_space

        # Coverage estimate
        coverage = total_autonomous / max(1, self.ESTIMATED_TOTAL_FEATURE_SPACE)
        coverage = min(1.0, coverage)

        return BoundarySurface(
            total_autonomous_features=total_autonomous,
            recombination_space=recombination_space,
            timescale_space=timescale_space,
            total_transforms=n_transforms,
            total_data_streams=n_streams,
            coverage_estimate=coverage,
            details={
                "n_timescales": n_timescales,
                "base_features": base_features,
            },
        )

    def expand_boundary(
        self,
        addition_type: str,
        addition: Any,
    ) -> BoundaryExpansion:
        """Expand the discovery boundary (inflating the balloon).

        When a human adds a new transform: everything computable from
        existing streams + new transform becomes Level 1.

        When a human adds a new data stream: everything computable from
        new stream + existing transforms becomes Level 1.

        Args:
            addition_type: "transform" or "data_stream"
            addition: TransformSpec or DataStreamSpec

        Returns:
            BoundaryExpansion describing what became AUTONOMOUS
        """
        before = self.characterize_boundary_surface()

        if addition_type == "transform" and isinstance(addition, TransformSpec):
            self.library.add_transform(addition)

            # New Level 1 features: this transform × all existing streams
            streams = self.library.get_data_streams()
            new_features = [
                f"{addition.name}({s})" for s in streams
            ]

            after = self.characterize_boundary_surface()
            return BoundaryExpansion(
                expansion_type="transform",
                addition_name=addition.name,
                new_autonomous_features=after.total_autonomous_features - before.total_autonomous_features,
                description=(
                    f"Adding transform '{addition.name}' expanded boundary by "
                    f"{after.total_autonomous_features - before.total_autonomous_features} "
                    f"autonomous features"
                ),
                new_level_1_features=new_features,
            )

        elif addition_type == "data_stream" and isinstance(addition, DataStreamSpec):
            self.library.add_data_stream(addition)

            # New Level 1 features: all existing transforms × this stream
            transforms = self.library.get_transforms()
            new_features = [
                f"{t}({addition.name})" for t in transforms
            ]

            after = self.characterize_boundary_surface()
            return BoundaryExpansion(
                expansion_type="data_stream",
                addition_name=addition.name,
                new_autonomous_features=after.total_autonomous_features - before.total_autonomous_features,
                description=(
                    f"Adding data stream '{addition.name}' expanded boundary by "
                    f"{after.total_autonomous_features - before.total_autonomous_features} "
                    f"autonomous features"
                ),
                new_level_1_features=new_features,
            )

        return BoundaryExpansion(
            expansion_type=addition_type,
            description="No expansion (invalid addition type)",
        )

    def generate_human_research_brief(
        self,
        signature: Any = None,
        classification: Any = None,
    ) -> ResearchBrief:
        """Generate a structured research brief for human researchers.

        For gaps classified as DIRECTED or CREATIVE, produce a document
        containing quantitative gap description, what's been tried, what's
        missing, search directions, and priority.

        Args:
            signature: AnomalySignature (L0 output)
            classification: GapClassification (L1 output)

        Returns:
            ResearchBrief with all relevant information
        """
        brief = ResearchBrief()

        # Populate from signature if available
        if signature is not None:
            brief.gap_description = getattr(signature, 'summary', lambda: "Gap detected")()
            brief.severity = f"{getattr(signature, 'n_misses', 0)} missed trades"

            # Temporal profile
            temporal = getattr(signature, 'temporal_profile', None)
            if temporal is not None:
                peak_hours = getattr(temporal, 'peak_hours', [])
                if peak_hours:
                    brief.temporal_profile = f"Peak hours: {peak_hours}"

            # Regime context
            regime = getattr(signature, 'regime_profile', None)
            if regime is not None:
                regime_counts = getattr(regime, 'regime_counts', {})
                if regime_counts:
                    brief.regime_context = f"Regime distribution: {dict(regime_counts)}"

        # Populate from classification if available
        if classification is not None:
            top_features = getattr(classification, 'top_features', [])
            brief.features_tried = top_features if top_features else []
            brief.missing_signal_type = getattr(classification, 'gap_type', 'unknown')
            if hasattr(brief.missing_signal_type, 'value'):
                brief.missing_signal_type = brief.missing_signal_type.value

        # Add search directions
        if signature is not None:
            lead_lag = getattr(signature, 'lead_lag_profile', None)
            if lead_lag is not None:
                leads = getattr(lead_lag, 'leading_instruments', [])
                if leads:
                    brief.search_directions.append(
                        f"Cross-asset signal from: {leads}"
                    )

            temporal = getattr(signature, 'temporal_profile', None)
            if temporal is not None:
                has_pattern = getattr(temporal, 'has_strong_pattern', False)
                if has_pattern:
                    brief.search_directions.append(
                        "Strong temporal pattern — investigate event-driven features"
                    )

        if not brief.search_directions:
            brief.search_directions.append(
                "General exploration recommended — no strong directional signal"
            )

        # Priority from severity
        n_misses = getattr(signature, 'n_misses', 0) if signature else 0
        if n_misses > 50:
            brief.priority = "HIGH"
        elif n_misses > 20:
            brief.priority = "MEDIUM"
        else:
            brief.priority = "LOW"

        return brief
