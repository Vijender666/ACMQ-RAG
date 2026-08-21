"""
memory_types.py

Core data structures for ACMQ-RAG (Adaptive Memory Construction Framework
for Graph-Based Retrieval-Augmented Generation).

This module defines the MemoryUnit and related data models used throughout
the memory construction pipeline.

Author: Vijender Reddy
Project: ACMQ-RAG
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ----------------------------------------------------------------------
# Memory Decision Types
# ----------------------------------------------------------------------

class MemoryDecision(Enum):
    """
    Final decision assigned by the Memory Decision Engine.
    """

    ACCEPT = "accept"
    MERGE = "merge"
    UPDATE = "update"
    ARCHIVE = "archive"
    DISCARD = "discard"


# ----------------------------------------------------------------------
# Source Metadata
# ----------------------------------------------------------------------

@dataclass
class SourceMetadata:
    """
    Stores provenance information for a knowledge unit.
    """

    source_id: Optional[str] = None
    source_name: Optional[str] = None
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None

    timestamp: Optional[str] = None

    author: Optional[str] = None

    reliability: float = 1.0


# ----------------------------------------------------------------------
# Quality Score
# ----------------------------------------------------------------------

@dataclass
class QualityScore:
    """
    Stores all quality dimensions calculated by CMQM.
    """

    novelty: float = 0.0

    semantic_consistency: float = 0.0

    source_reliability: float = 0.0

    content_quality: float = 0.0

    graph_importance: float = 0.0

    retrieval_utility: float = 0.0

    temporal_freshness: float = 0.0

    overall_score: float = 0.0

    evaluated: bool = False


# ----------------------------------------------------------------------
# Memory Unit
# ----------------------------------------------------------------------

@dataclass
class MemoryUnit:
    """
    Represents one candidate memory before entering the knowledge graph.

    Every OpenIE triple is converted into a MemoryUnit.

    Example
    -------
    Subject:
        Steve Jobs

    Relation:
        founded

    Object:
        Apple
    """

    # -----------------------------
    # Triple
    # -----------------------------

    subject: str

    relation: str

    object: str

    # -----------------------------
    # Original Information
    # -----------------------------

    original_text: Optional[str] = None

    normalized_text: Optional[str] = None

    # -----------------------------
    # Source Metadata
    # -----------------------------

    metadata: SourceMetadata = field(default_factory=SourceMetadata)

    # -----------------------------
    # CMQM Scores
    # -----------------------------

    quality: QualityScore = field(default_factory=QualityScore)

    # -----------------------------
    # Final Memory Decision
    # -----------------------------

    decision: Optional[MemoryDecision] = None

    # -----------------------------
    # ACMQ-RAG Evolution Tracking
    # -----------------------------

    matched_memory_hash: Optional[str] = None

    previous_decision: Optional[MemoryDecision] = None

    decision_reason: Optional[str] = None

    # -----------------------------
    # Graph Information
    # -----------------------------

    graph_node_ids: List[int] = field(default_factory=list)

    connected_entities: List[str] = field(default_factory=list)

    # -----------------------------
    # Additional Information
    # -----------------------------

    tags: List[str] = field(default_factory=list)

    attributes: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def triple(self):
        """
        Returns the triple representation.
        """

        return (
            self.subject,
            self.relation,
            self.object
        )

    # ------------------------------------------------------------------

    def as_text(self):
        """
        Returns readable triple.
        """

        return f"{self.subject} --[{self.relation}]--> {self.object}"

    # ------------------------------------------------------------------

    def is_evaluated(self):
        """
        Returns whether CMQM has evaluated this memory.
        """

        return self.quality.evaluated

    # ------------------------------------------------------------------

    def mark_evaluated(self):
        """
        Marks this memory as evaluated.
        """

        self.quality.evaluated = True

    # ------------------------------------------------------------------

    def set_decision(self, decision: MemoryDecision):
        """
        Assign memory decision.
        """

        self.decision = decision

    # ------------------------------------------------------------------

    def __str__(self):

        return (
            f"MemoryUnit("
            f"{self.subject}, "
            f"{self.relation}, "
            f"{self.object}, "
            f"Score={self.quality.overall_score:.3f}, "
            f"Decision={self.decision}"
            f")"
        )