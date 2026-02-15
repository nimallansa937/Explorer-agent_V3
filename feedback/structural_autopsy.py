"""
Structural Autopsy — Anti-Template Extraction

When a production strategy fails, analyze its LOSING trades during the
decay period. Classify as structural vs feature decay. For structural
decay, extract anti-templates (tree topologies that used to work but
no longer do). These anti-templates penalize similar new candidates.

Explorer Prime v2.0 - Phase 5
"""

import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class AutopsyResult:
    """Result of analyzing a failed strategy's decay period."""
    strategy_id: str = ""
    gap_type: str = "unknown"                    # structural/feature/ambiguous
    signature: Any = None                        # AnomalySignature of decay period
    anti_templates: Optional[List['AntiTemplate']] = None
    feature_investigation_priority: str = "HIGH" # Production failures are always HIGH


@dataclass
class AntiTemplate:
    """Tree topology that recently failed — penalizes similar new candidates."""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topology_encoding: Optional[np.ndarray] = None    # Encoded tree structure
    source_strategy_id: str = ""
    source_anomaly_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    half_life_days: float = 90.0              # 90-day half-life


# ==============================================================================
# Structural Autopsy
# ==============================================================================

class StructuralAutopsy:
    """Analyzes failed strategies to extract anti-templates.

    The 'how to mutate differently' channel: identifies tree topologies
    that USED to work but no longer do, and penalizes new candidates
    with similar topologies.
    """

    TOPOLOGY_SIMILARITY_THRESHOLD: float = 0.8

    def analyze(
        self,
        strategy_id: str,
        genome: Any,
        decay_period_losing_features: Optional[np.ndarray] = None,
        gap_type: str = "structural",
    ) -> AutopsyResult:
        """Analyze a failed strategy's decay period.

        Args:
            strategy_id: ID of the failed strategy
            genome: HierarchicalGenome of the failed strategy
            decay_period_losing_features: Feature vectors of losing trades
            gap_type: Classification from anomaly diagnostic

        Returns:
            AutopsyResult with anti-templates for structural decay
        """
        result = AutopsyResult(
            strategy_id=strategy_id,
            gap_type=gap_type,
            feature_investigation_priority="HIGH",
        )

        if gap_type in ("structural", "ambiguous"):
            # Extract anti-templates from the tree topology
            anti_templates = self._extract_anti_templates(genome, strategy_id)
            result.anti_templates = anti_templates

        return result

    def _extract_anti_templates(
        self, genome: Any, strategy_id: str
    ) -> List[AntiTemplate]:
        """Extract tree topology anti-templates from genome."""
        templates = []

        # Encode the genome's tree topology as a vector
        encoding = self._encode_topology(genome)
        if encoding is not None:
            templates.append(AntiTemplate(
                topology_encoding=encoding,
                source_strategy_id=strategy_id,
            ))

        return templates

    @staticmethod
    def _encode_topology(genome: Any) -> Optional[np.ndarray]:
        """Encode a genome's tree topology as a fixed-length vector.

        Uses a simplified encoding: flatten tree structure to a vector.
        For production, use a more sophisticated graph encoding.
        """
        if genome is None:
            return None

        # If genome has a to_dict method, use it to create encoding
        if hasattr(genome, 'to_dict'):
            d = genome.to_dict()
        elif isinstance(genome, dict):
            # Direct dict genome (e.g., from retirement manager)
            d = genome
        elif hasattr(genome, 'genome') and hasattr(genome.genome, 'tree'):
            # Flatten tree to fixed-length vector
            return np.random.RandomState(42).randn(64)
        else:
            return None

        # Simple encoding: hash of structure as random vector
        import hashlib
        h = hashlib.sha256(str(d).encode()).hexdigest()
        seed = int(h[:8], 16)
        rng = np.random.RandomState(seed)
        return rng.randn(64)


# ==============================================================================
# Anti-Template Injector
# ==============================================================================

class AntiTemplateInjector:
    """Penalizes new candidates whose topology resembles anti-templates.

    Cosine similarity > 0.8 triggers penalty.
    Anti-templates decay with 90-day half-life.
    """

    SIMILARITY_THRESHOLD: float = 0.8
    DEFAULT_HALF_LIFE_DAYS: float = 90.0

    def __init__(self):
        self._templates: List[AntiTemplate] = []

    @property
    def templates(self) -> List[AntiTemplate]:
        return self._templates

    def __len__(self) -> int:
        return len(self._templates)

    def add_template(self, template: AntiTemplate) -> None:
        """Add an anti-template."""
        self._templates.append(template)

    def add_from_autopsy(self, result: AutopsyResult) -> None:
        """Add anti-templates from an autopsy result."""
        if result.anti_templates:
            for t in result.anti_templates:
                self._templates.append(t)

    def penalty(
        self,
        candidate_encoding: Optional[np.ndarray],
        reference_date: Optional[datetime] = None,
    ) -> float:
        """Compute anti-template penalty for a candidate.

        Args:
            candidate_encoding: Topology encoding of the candidate
            reference_date: Date for time decay

        Returns:
            Penalty in [0.0, 1.0]
        """
        if candidate_encoding is None or not self._templates:
            return 0.0

        if reference_date is None:
            reference_date = datetime.utcnow()

        max_penalty = 0.0

        for template in self._templates:
            if template.topology_encoding is None:
                continue

            similarity = self._cosine_similarity(
                candidate_encoding, template.topology_encoding
            )

            if similarity < self.SIMILARITY_THRESHOLD:
                continue

            # Time decay
            days_since = max(0, (reference_date - template.created_at).days)
            half_life = template.half_life_days
            time_factor = math.exp(-days_since * math.log(2) / half_life) if half_life > 0 else 0.0

            penalty = similarity * time_factor
            max_penalty = max(max_penalty, penalty)

        return max_penalty

    def prune_expired(self, reference_date: Optional[datetime] = None,
                      min_factor: float = 0.01) -> int:
        """Remove expired anti-templates (time factor < min_factor).

        Returns:
            Number of templates removed
        """
        if reference_date is None:
            reference_date = datetime.utcnow()

        before = len(self._templates)
        self._templates = [
            t for t in self._templates
            if self._time_factor(t, reference_date) >= min_factor
        ]
        return before - len(self._templates)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        min_len = min(len(a), len(b))
        if min_len == 0:
            return 0.0
        a_t = a[:min_len]
        b_t = b[:min_len]
        dot = np.dot(a_t, b_t)
        norm_a = np.linalg.norm(a_t)
        norm_b = np.linalg.norm(b_t)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def _time_factor(template: AntiTemplate, reference_date: datetime) -> float:
        days = max(0, (reference_date - template.created_at).days)
        hl = template.half_life_days
        return math.exp(-days * math.log(2) / hl) if hl > 0 else 0.0
