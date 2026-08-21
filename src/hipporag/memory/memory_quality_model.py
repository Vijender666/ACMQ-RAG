"""
memory_quality_model.py

Context-Aware Memory Quality Model (CMQM)

Evaluates MemoryUnits using six quality dimensions:

1. Novelty
2. Semantic Consistency
3. Source Reliability
4. Graph Structural Importance
5. Retrieval Utility
6. Temporal Freshness

ACMQ-RAG
"""

from typing import List
from datetime import datetime

import numpy as np
import sentence_transformers

from .memory_types import MemoryUnit
from ..utils.misc_utils import compute_mdhash_id


class MemoryQualityModel:
    """
    Context-Aware Memory Quality Model (CMQM).

    Evaluates memory quality using semantic, graph,
    retrieval, source, and temporal signals.
    """

    def __init__(
        self,
        novelty_weight: float = 0.15,
        consistency_weight: float = 0.15,
        reliability_weight: float = 0.15,
        content_quality_weight: float = 0.15,
        importance_weight: float = 0.15,
        utility_weight: float = 0.15,
        freshness_weight: float = 0.10,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        duplicate_threshold: float = 0.85,
        graph_stats=None,
    ):

        self.weights = {
            "novelty": novelty_weight,
            "consistency": consistency_weight,
            "reliability": reliability_weight,
            "content_quality": content_quality_weight,
            "importance": importance_weight,
            "utility": utility_weight,
            "freshness": freshness_weight,
        }

        self.duplicate_threshold = duplicate_threshold

        # Graph statistics supplied by HippoRAG
        self.graph_stats = graph_stats or {}

        # BGE embedding model
        self.embedding_model = (
            sentence_transformers.SentenceTransformer(
                embedding_model
            )
        )

    # =========================================================
    # MAIN EVALUATION
    # =========================================================

    def evaluate(
        self,
        memory: MemoryUnit,
        existing_memories: List[MemoryUnit],
    ) -> MemoryUnit:

        memory.quality.novelty = self.compute_novelty(
            memory,
            existing_memories,
        )

        memory.quality.semantic_consistency = (
            self.compute_semantic_consistency(
                memory,
                existing_memories,
            )
        )

        memory.quality.source_reliability = (
            self.compute_source_reliability(memory)
        )

        memory.quality.content_quality = (
            self.compute_content_quality(memory)
        )

        memory.quality.graph_importance = (
            self.compute_graph_importance(memory)
        )

        memory.quality.retrieval_utility = (
            self.compute_retrieval_utility(memory)
        )

        memory.quality.temporal_freshness = (
            self.compute_temporal_freshness(memory)
        )

        memory.quality.overall_score = (
            self.compute_overall_score(memory)
        )

        memory.mark_evaluated()

        return memory

    # =========================================================
    # BATCH EVALUATION
    # =========================================================

    def evaluate_batch(
        self,
        memories: List[MemoryUnit],
        existing_memories: List[MemoryUnit],
    ) -> List[MemoryUnit]:

        results = []

        for memory in memories:
            results.append(
                self.evaluate(
                    memory,
                    existing_memories,
                )
            )

        return results

    # =========================================================
    # MEMORY TEXT
    # =========================================================

    def memory_to_text(
        self,
        memory: MemoryUnit,
    ) -> str:
        """
        Convert a MemoryUnit into semantic text.
        """

        return (
            f"{memory.subject} "
            f"{memory.relation} "
            f"{memory.object}"
        )

    # =========================================================
    # STEP 1
    # SEMANTIC NOVELTY
    # =========================================================

    def compute_novelty(
        self,
        memory: MemoryUnit,
        existing_memories: List[MemoryUnit],
    ) -> float:
        """
        Compute semantic novelty using BGE embeddings.

        Higher similarity to an existing memory means
        the candidate is less novel.

        Returns:
            1.0 -> completely novel
            0.0 -> semantic duplicate
        """

        if not existing_memories:
            return 1.0

        candidate_text = self.memory_to_text(memory)

        existing_texts = [
            self.memory_to_text(existing)
            for existing in existing_memories
        ]

        candidate_embedding = self.embedding_model.encode(
            candidate_text,
            normalize_embeddings=True,
        )

        existing_embeddings = self.embedding_model.encode(
            existing_texts,
            normalize_embeddings=True,
        )

        similarities = np.dot(
            existing_embeddings,
            candidate_embedding,
        )

        max_similarity = float(
            np.max(similarities)
        )

        novelty = 1.0 - max_similarity

        return round(
            max(0.0, min(1.0, novelty)),
            4,
        )

    # =========================================================
    # SEMANTIC CONSISTENCY
    # =========================================================

    def compute_semantic_consistency(
        self,
        memory: MemoryUnit,
        existing_memories: List[MemoryUnit],
    ) -> float:
        """
        Basic semantic consistency score.

        This can later be upgraded with conflict detection
        or NLI-based consistency checking.
        """

        return 1.0

    # =========================================================
    # CONTENT QUALITY
    # =========================================================

    def compute_content_quality(
        self,
        memory: MemoryUnit,
    ) -> float:
        """
        Estimate basic factual/content validity.

        Reject obviously malformed placeholder memories
        such as:
            unknown | unknown | unknown
        """

        values = [
            str(memory.subject).strip().lower(),
            str(memory.relation).strip().lower(),
            str(memory.object).strip().lower(),
        ]

        invalid_values = {
            "",
            "unknown",
            "none",
            "null",
            "n/a",
            "na",
            "?",
        }

        if any(value in invalid_values for value in values):
            return 0.0

        return 1.0

    # =========================================================
    # SOURCE RELIABILITY
    # =========================================================

    def compute_source_reliability(
        self,
        memory: MemoryUnit,
    ) -> float:

        return float(
            memory.metadata.reliability
        )

    # =========================================================
    # GRAPH STRUCTURAL IMPORTANCE
    # =========================================================

    def compute_graph_importance(
        self,
        memory: MemoryUnit,
    ) -> float:
        """
        Compute graph-aware importance using normalized
        node degree and PageRank.
        """

        if not self.graph_stats:
            return 1.0

        def resolve_entity_key(value):
            """
            Resolve a memory entity to the canonical HippoRAG
            graph-node key.

            Important:
            - Preserve original entity text when hashing.
            - HippoRAG graph IDs are generated from the original
              entity text, not lowercased text.
            """

            original_value = str(value).strip()
            normalized_value = original_value.lower()

            # Direct graph-node ID
            if normalized_value in self.graph_stats:
                return normalized_value

            # Already a HippoRAG entity ID
            if normalized_value.startswith("entity-"):
                return normalized_value

            # Raw entity text -> canonical HippoRAG entity ID.
            # DO NOT lowercase before hashing.
            return str(
                compute_mdhash_id(
                    content=original_value,
                    prefix="entity-",
                )
            ).lower()

        subject_key = resolve_entity_key(memory.subject)
        object_key = resolve_entity_key(memory.object)

        default_stats = {
            "degree": 0.0,
            "pagerank": 0.0,
        }

        subject_stats = self.graph_stats.get(
            subject_key,
            default_stats,
        )

        object_stats = self.graph_stats.get(
            object_key,
            default_stats,
        )

        subject_importance = (
            0.5 * float(subject_stats.get("degree", 0.0))
            +
            0.5 * float(subject_stats.get("pagerank", 0.0))
        )

        object_importance = (
            0.5 * float(object_stats.get("degree", 0.0))
            +
            0.5 * float(object_stats.get("pagerank", 0.0))
        )

        importance = max(
            subject_importance,
            object_importance,
        )

        return round(
            max(
                0.0,
                min(1.0, importance),
            ),
            4,
        )

    # =========================================================
    # RETRIEVAL UTILITY
    # =========================================================

    def compute_retrieval_utility(
        self,
        memory: MemoryUnit,
    ) -> float:
        """
        Compute retrieval utility using:

        1. Entity coverage
        2. Graph connectivity
        3. Graph centrality

        Entity hashing preserves the original entity text so that
        the generated IDs match HippoRAG graph-node IDs.
        """

        if not self.graph_stats:
            return 1.0

        def resolve_entity_key(value):
            original_value = str(value).strip()
            normalized_value = original_value.lower()

            # Direct graph-node ID
            if normalized_value in self.graph_stats:
                return normalized_value

            # Already a HippoRAG entity ID
            if normalized_value.startswith("entity-"):
                return normalized_value

            # Raw entity text -> canonical HippoRAG entity ID.
            # IMPORTANT: preserve original casing before hashing.
            return str(
                compute_mdhash_id(
                    content=original_value,
                    prefix="entity-",
                )
            ).lower()

        subject_key = resolve_entity_key(memory.subject)
        object_key = resolve_entity_key(memory.object)

        default_stats = {
            "degree": 0.0,
            "pagerank": 0.0,
        }

        subject_stats = self.graph_stats.get(
            subject_key,
            default_stats,
        )

        object_stats = self.graph_stats.get(
            object_key,
            default_stats,
        )

        subject_exists = subject_key in self.graph_stats
        object_exists = object_key in self.graph_stats

        entity_coverage = (
            int(subject_exists)
            +
            int(object_exists)
        ) / 2.0

        subject_degree = float(
            subject_stats.get("degree", 0.0)
        )

        object_degree = float(
            object_stats.get("degree", 0.0)
        )

        graph_connectivity = (
            subject_degree +
            object_degree
        ) / 2.0

        subject_pagerank = float(
            subject_stats.get("pagerank", 0.0)
        )

        object_pagerank = float(
            object_stats.get("pagerank", 0.0)
        )

        graph_centrality = (
            subject_pagerank +
            object_pagerank
        ) / 2.0

        utility = (
            0.40 * entity_coverage
            +
            0.35 * graph_connectivity
            +
            0.25 * graph_centrality
        )

        return round(
            max(
                0.0,
                min(1.0, utility),
            ),
            4,
        )

    # =========================================================
    # TEMPORAL FRESHNESS
    # =========================================================

    def compute_temporal_freshness(
        self,
        memory: MemoryUnit,
    ) -> float:
        """
        Compute temporal freshness using exponential time decay.

        A 30-day half-life is used:

            freshness = 2^(-age_days / 30)

        Therefore:
            0 days  -> 1.000
            30 days -> 0.500
            60 days -> 0.250
            90 days -> 0.125

        Memories without a valid timestamp receive a neutral
        freshness value of 1.0 for backward compatibility.
        """

        timestamp = memory.metadata.timestamp

        if not timestamp:
            return 1.0

        try:
            memory_time = datetime.fromisoformat(
                timestamp
            )

            now = datetime.utcnow()

            age_seconds = (
                now - memory_time
            ).total_seconds()

            age_days = max(
                0.0,
                age_seconds / 86400.0,
            )

            half_life_days = 30.0

            freshness = (
                2.0 ** (
                    -age_days / half_life_days
                )
            )

            return round(
                max(
                    0.0,
                    min(1.0, freshness),
                ),
                4,
            )

        except (ValueError, TypeError):
            return 1.0

    # =========================================================
    # OVERALL QUALITY SCORE
    # =========================================================

    def compute_overall_score(
        self,
        memory: MemoryUnit,
    ) -> float:
        """
        Calculate weighted overall memory quality score.
        """

        q = memory.quality

        score = (

            q.novelty
            * self.weights["novelty"]

            +

            q.semantic_consistency
            * self.weights["consistency"]

            +

            q.source_reliability
            * self.weights["reliability"]

            +

            q.content_quality
            * self.weights["content_quality"]

            +

            q.graph_importance
            * self.weights["importance"]

            +

            q.retrieval_utility
            * self.weights["utility"]

            +

            q.temporal_freshness
            * self.weights["freshness"]

        )

        return round(
            score,
            4,
        )
