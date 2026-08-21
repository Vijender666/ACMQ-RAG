"""
memory_adapter.py

Adapter between HippoRAG OpenIE triples and ACMQ-RAG Memory Pipeline.

Converts extracted OpenIE triples into MemoryUnit objects
that can be evaluated by the Memory Quality Model.

Author: Vijender Reddy
Project: ACMQ-RAG
"""

from typing import List, Tuple

from .memory_types import MemoryUnit, SourceMetadata


class MemoryAdapter:
    """
    Converts HippoRAG triples into MemoryUnit objects.
    """

    @staticmethod
    def triple_to_memory(
        triple: Tuple[str, str, str]
    ) -> MemoryUnit:
        """
        Convert one OpenIE triple into a MemoryUnit.
        """

        if len(triple) != 3:
            raise ValueError(
                f"Expected triple of length 3, got {len(triple)}"
            )

        subject, relation, obj = triple

        return MemoryUnit(
            subject=str(subject).strip(),
            relation=str(relation).strip(),
            object=str(obj).strip(),
        )

    # -------------------------------------------------------------

    @staticmethod
    def triples_to_memories(
        triples: List[Tuple[str, str, str]]
    ) -> List[MemoryUnit]:
        """
        Convert a list of triples into MemoryUnits.
        """

        memories = []

        for triple in triples:
            try:
                memories.append(
                    MemoryAdapter.triple_to_memory(triple)
                )
            except Exception:
                continue

        return memories

    # -------------------------------------------------------------

    @staticmethod
    def chunk_triples_to_memories(
        chunk_ids: List[str],
        chunk_triples: List[List[Tuple[str, str, str]]]
    ) -> List[MemoryUnit]:
        """
        Convert chunk-wise triples into MemoryUnits while preserving chunk IDs.
        """

        memories = []

        for chunk_id, triples in zip(chunk_ids, chunk_triples):

            for triple in triples:

                if len(triple) != 3:
                    continue

                subject, relation, obj = triple

                memories.append(
                    MemoryUnit(
                        subject=str(subject).strip(),
                        relation=str(relation).strip(),
                        object=str(obj).strip(),
                        metadata=SourceMetadata(
                            chunk_id=chunk_id
                        )
                    )
                )
        return memories

    # -------------------------------------------------------------

    @staticmethod
    def memories_to_triples(
        memories: List[MemoryUnit]
    ) -> List[Tuple[str, str, str]]:
        """
        Convert MemoryUnits back into triples.
        """

        triples = []

        for memory in memories:
            triples.append(
                (
                    memory.subject,
                    memory.relation,
                    memory.object,
                )
            )

        return triples

    # -------------------------------------------------------------

    @staticmethod
    def print_summary(
        memories: List[MemoryUnit]
    ):
        """
        Print adapter statistics.
        """

        print()
        print("=" * 55)
        print("ACMQ-RAG Memory Adapter")
        print("=" * 55)
        print(f"Converted Memories : {len(memories)}")
        print("=" * 55)
        print()