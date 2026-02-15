"""
Feedback Package for EXPLORER PRIME v2.0

Three-channel production feedback loop:
1. FailureArchive: Behavioral similarity-based negative seeding
2. StructuralAutopsy: Anti-template extraction from failed strategies
3. MetaLearning: Pipeline parameter calibration from failure distributions
"""

from .failure_archive import (
    FailureRecord,
    FailureArchive,
    FailureDistribution,
)

from .structural_autopsy import (
    AutopsyResult,
    AntiTemplate,
    StructuralAutopsy,
    AntiTemplateInjector,
)

from .meta_learning import (
    BimodalAnalysis,
    PipelineCalibration,
    MetaLearningSignal,
)

__all__ = [
    'FailureRecord',
    'FailureArchive',
    'FailureDistribution',
    'AutopsyResult',
    'AntiTemplate',
    'StructuralAutopsy',
    'AntiTemplateInjector',
    'BimodalAnalysis',
    'PipelineCalibration',
    'MetaLearningSignal',
]
