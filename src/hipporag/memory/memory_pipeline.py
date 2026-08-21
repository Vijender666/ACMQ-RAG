"""
memory_pipeline.py

Adaptive Memory Construction Pipeline (ACMQ-RAG)
"""

from typing import List, Tuple, Union

from .memory_types import MemoryUnit
from .memory_types import SourceMetadata
from .memory_quality_model import MemoryQualityModel
from .memory_decision_engine import MemoryDecisionEngine
from .memory_store import MemoryStore


class MemoryPipeline:
    """
    Adaptive Memory Construction Pipeline.
    """

    def __init__(self, graph_stats=None):

        self.graph_stats = graph_stats or {}

        self.quality_model = MemoryQualityModel(
            graph_stats=self.graph_stats
        )

        self.decision_engine = MemoryDecisionEngine()
        self.memory_store = MemoryStore()

    # -------------------------------------------------------------
    # TRIPLE -> MEMORY UNIT
    # -------------------------------------------------------------

    def triples_to_memory_units(
        self,
        triples: List[Tuple[str, str, str]],
        source_name: str = "Unknown",
        chunk_id: str = None,
        source_reliability: float = 1.0,
    ) -> List[MemoryUnit]:

        memory_units = []

        for triple in triples:

            subject, relation, obj = triple

            memory = MemoryUnit(
                subject=subject,
                relation=relation,
                object=obj,
                metadata=SourceMetadata(
                    source_name=source_name,
                    chunk_id=chunk_id,
                    reliability=max(
                        0.0,
                        min(1.0, float(source_reliability)),
                    ),
                ),
            )

            memory_units.append(memory)

        return memory_units

    # -------------------------------------------------------------
    # EVALUATE ONLY
    # -------------------------------------------------------------

    def evaluate_only(
        self,
        memory_units: List[MemoryUnit],
    ) -> List[MemoryUnit]:
        """
        Evaluate memory quality without making decisions
        or storing memories.

        Used by ACMQ-RAG when graph statistics must be
        available before the final memory decision.
        """

        existing_memories = self.memory_store.get_all()

        results = self.quality_model.evaluate_batch(
            memory_units,
            existing_memories,
        )

        print("\nACMQ-RAG: Memory Quality Scores")
        print("-" * 70)

        for i, memory in enumerate(results, 1):
            q = memory.quality

            print(
                f"{i:02d}. "
                f"{memory.subject} | "
                f"{memory.relation} | "
                f"{memory.object}"
            )

            print(
                f"    novelty={q.novelty:.3f} "
                f"consistency={q.semantic_consistency:.3f} "
                f"reliability={q.source_reliability:.3f} "
                f"importance={q.graph_importance:.3f} "
                f"utility={q.retrieval_utility:.3f} "
                f"freshness={q.temporal_freshness:.3f} "
                f"OVERALL={q.overall_score:.3f}"
            )

        print("-" * 70)

        return results

    # -------------------------------------------------------------
    # EVALUATE
    # -------------------------------------------------------------

    def evaluate(
        self,
        memory_units: List[MemoryUnit],
    ) -> List[MemoryUnit]:

        existing_memories = self.memory_store.get_all()

        return self.quality_model.evaluate_batch(
            memory_units,
            existing_memories,
        )

    # -------------------------------------------------------------
    # DECIDE
    # -------------------------------------------------------------

    def decide(
        self,
        memory_units: List[MemoryUnit],
    ) -> List[MemoryUnit]:
        """
        Apply the Adaptive Memory Decision Engine.
        """

        existing_memories = self.memory_store.get_all()

        return self.decision_engine.process_batch(
            memory_units,
            existing_memories,
        )

    # -------------------------------------------------------------
    # STORE
    # -------------------------------------------------------------

    def store(
        self,
        memory_units: List[MemoryUnit],
    ):

        self.memory_store.add_batch(
            memory_units
        )

    # -------------------------------------------------------------
    # COMPLETE PROCESS
    # -------------------------------------------------------------

    def process(
        self,
        memories: Union[
            List[MemoryUnit],
            List[Tuple[str, str, str]],
        ],
        source_name: str = "Unknown",
        chunk_id: str = None,
    ):
        """
        Complete ACMQ-RAG pipeline.

        Supports:

        1. Raw OpenIE triples
        2. MemoryUnit objects

        Returns:

            MemoryStore
        """

        # Empty input
        if len(memories) == 0:
            return self.memory_store

        # Already MemoryUnits
        if isinstance(memories[0], MemoryUnit):

            memory_units = memories

        # Raw triples
        else:

            memory_units = self.triples_to_memory_units(
                memories,
                source_name,
                chunk_id,
            )

        # Quality evaluation
        memory_units = self.evaluate(
            memory_units
        )

        # Decision engine
        memory_units = self.decide(
            memory_units
        )

        # Store
        self.store(
            memory_units
        )

        return self.memory_store

    # -------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------

    def statistics(self):

        return self.memory_store.statistics()

    # -------------------------------------------------------------

    def print_statistics(self):

        self.memory_store.print_statistics()

    # -------------------------------------------------------------

    def clear(self):

        self.memory_store.clear()
