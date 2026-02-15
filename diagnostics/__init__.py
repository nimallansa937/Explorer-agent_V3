"""
Diagnostics Package for EXPLORER PRIME v2.0

Provides anomaly detection, gap classification, intervention protocols,
and discovery boundary formalization.

Key Components:
- AnomalySignature: Structured L0 gap characterization
- AnomalyDiagnostic: Missed-trade analysis + Random Forest gap classification
- InterventionProtocol: Sequential structure-first intervention (Phase 4)
- DiscoveryBoundary: Four-level discovery taxonomy (Phase 8)
"""

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

from .anomaly_diagnostic import (
    AnomalyDiagnostic,
)

from .intervention_protocol import (
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

from .discovery_boundary import (
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
)

__all__ = [
    # Phase 3: Anomaly Diagnostic
    'TradeOpportunity',
    'TemporalProfile',
    'RegimeProfile',
    'AssetProfile',
    'VolatilityProfile',
    'PrecedingPattern',
    'LeadLagProfile',
    'AnomalySignature',
    'GapType',
    'GapClassification',
    'DiagnosisResult',
    'AnomalyDiagnostic',

    # Phase 4: Intervention Protocol
    'InterventionType',
    'PlanStatus',
    'Phase1Decision',
    'InterventionPlan',
    'Phase1Result',
    'FeatureProposal',
    'TreeSeed',
    'InterventionRouter',
    'DirectedFeatureScout',
    'StructuralSeedInjector',

    # Phase 8: Discovery Boundary
    'DiscoveryLevel',
    'CapabilityCategory',
    'DiscoveryBoundary',
    'ComputationalLibrary',
    'TransformSpec',
    'DataStreamSpec',
    'DiscoveryClassification',
    'BoundarySurface',
    'BoundaryExpansion',
    'ResearchBrief',
]
