"""
memory_store.py

Adaptive Memory Store for ACMQ-RAG.

Responsible for maintaining evolving long-term memory.

MERGE:
    Combine new evidence with an existing memory.

UPDATE:
    Replace an older conflicting memory when the new
    memory has stronger quality.

ARCHIVE:
    Preserve the memory but exclude it from active graph
    construction.

ACCEPT:
    Add a new high-quality memory.

DISCARD:
    Do not store the memory.
"""

from typing import Dict, List, Optional

from .memory_types import MemoryDecision, MemoryUnit
from .memory_utils import generate_memory_hash


class MemoryStore:
    """
    Adaptive memory repository.
    """

    def __init__(self):

        self.memories: Dict[str, MemoryUnit] = {}

        self.accepted: List[str] = []
        self.merged: List[str] = []
        self.updated: List[str] = []
        self.archived: List[str] = []
        self.discarded: List[str] = []

        # -----------------------------------------------------
        # ACMQ-RAG DECISION HISTORY
        # -----------------------------------------------------

        self.decision_history: List[dict] = []

        self.merge_history: List[dict] = []
        self.update_history: List[dict] = []
        self.archive_history: List[dict] = []
        self.discard_history: List[dict] = []
        self.accept_history: List[dict] = []

        # Lazy-loaded semantic embedding model
        self._embedding_model = None

    # ---------------------------------------------------------
    # DECISION HISTORY
    # ---------------------------------------------------------

    def _record_decision(self, memory: MemoryUnit):

        record = {
            "subject": memory.subject,
            "relation": memory.relation,
            "object": memory.object,
            "decision": (
                memory.decision.value
                if memory.decision
                else None
            ),
            "score": memory.quality.overall_score,
            "matched_memory_hash": (
                memory.matched_memory_hash
            ),
            "reason": memory.decision_reason,
        }

        self.decision_history.append(record)

        if memory.decision == MemoryDecision.ACCEPT:
            self.accept_history.append(record)

        elif memory.decision == MemoryDecision.MERGE:
            self.merge_history.append(record)

        elif memory.decision == MemoryDecision.UPDATE:
            self.update_history.append(record)

        elif memory.decision == MemoryDecision.ARCHIVE:
            self.archive_history.append(record)

        elif memory.decision == MemoryDecision.DISCARD:
            self.discard_history.append(record)

    # ---------------------------------------------------------
    # ADD
    # ---------------------------------------------------------

    def add(self, memory: MemoryUnit):

        self._record_decision(memory)

        if memory.decision == MemoryDecision.DISCARD:

            memory_hash = generate_memory_hash(memory)

            if memory_hash not in self.discarded:
                self.discarded.append(memory_hash)

            return memory

        if memory.decision == MemoryDecision.MERGE:
            return self._merge_memory(memory)

        if memory.decision == MemoryDecision.UPDATE:
            return self._update_memory(memory)

        if memory.decision == MemoryDecision.ARCHIVE:
            return self._archive_memory(memory)

        return self._accept_memory(memory)

    # ---------------------------------------------------------
    # ACCEPT
    # ---------------------------------------------------------

    def _accept_memory(
        self,
        memory: MemoryUnit,
    ):

        memory_hash = generate_memory_hash(memory)

        if memory_hash in self.memories:
            return self.memories[memory_hash]

        self.memories[memory_hash] = memory

        if memory_hash not in self.accepted:
            self.accepted.append(memory_hash)

        return memory

    # ---------------------------------------------------------
    # MERGE
    # ---------------------------------------------------------

    def _merge_memory(
        self,
        memory: MemoryUnit,
    ):
        """
        Merge a candidate memory only when it represents
        the same underlying fact.

        ACMQ-RAG merge policy:
            1. Exact hash match -> merge.
            2. Same subject + relation + object -> merge.
            3. Semantic similarity alone is NOT sufficient
               to replace a different entity/fact.
        """

        memory_hash = generate_memory_hash(memory)

        # -----------------------------------------------------
        # Exact existing memory
        # -----------------------------------------------------

        if memory_hash in self.memories:

            existing = self.memories[memory_hash]

            if (
                memory.quality.overall_score
                > existing.quality.overall_score
            ):
                existing.quality = memory.quality

            if memory.metadata.source_name:
                existing.metadata.source_name = (
                    memory.metadata.source_name
                )

            if memory.metadata.chunk_id:
                existing.metadata.chunk_id = (
                    memory.metadata.chunk_id
                )

            # MERGE is a mutually exclusive classification.
            self._remove_from_collections(memory_hash)

            if memory_hash not in self.merged:
                self.merged.append(memory_hash)

            return existing

        # -----------------------------------------------------
        # Structural fact match
        # -----------------------------------------------------

        subject = memory.subject.strip().lower()
        relation = memory.relation.strip().lower()
        object_value = memory.object.strip().lower()

        for existing in self.memories.values():

            same_subject = (
                existing.subject.strip().lower()
                == subject
            )

            same_relation = (
                existing.relation.strip().lower()
                == relation
            )

            same_object = (
                existing.object.strip().lower()
                == object_value
            )

            if (
                same_subject
                and same_relation
                and same_object
            ):

                existing_hash = generate_memory_hash(existing)

                if (
                    memory.quality.overall_score
                    > existing.quality.overall_score
                ):
                    existing.quality = memory.quality

                if memory.metadata.source_name:
                    existing.metadata.source_name = (
                        memory.metadata.source_name
                    )

                if memory.metadata.chunk_id:
                    existing.metadata.chunk_id = (
                        memory.metadata.chunk_id
                    )

                # MERGE is a mutually exclusive classification.
                self._remove_from_collections(existing_hash)

                if existing_hash not in self.merged:
                    self.merged.append(existing_hash)

                return existing

        # -----------------------------------------------------
        # No safe merge
        #
        # Do NOT use semantic similarity to replace a
        # different subject/entity.
        # -----------------------------------------------------

        return self._accept_memory(memory)

    # ---------------------------------------------------------
    # UPDATE
    # ---------------------------------------------------------

    def _update_memory(
        self,
        memory: MemoryUnit,
    ):

        target = self._find_same_subject_relation(memory)

        if target is None:
            return self._accept_memory(memory)

        old_hash = generate_memory_hash(target)
        new_hash = generate_memory_hash(memory)

        # Remove old memory
        if old_hash in self.memories:
            del self.memories[old_hash]

        # Remove old classification
        self._remove_from_collections(old_hash)

        # Store new memory
        self.memories[new_hash] = memory

        self._remove_from_collections(new_hash)

        if new_hash not in self.updated:
            self.updated.append(new_hash)

        return memory

    # ---------------------------------------------------------
    # ARCHIVE
    # ---------------------------------------------------------

    def _archive_memory(
        self,
        memory: MemoryUnit,
    ):

        memory_hash = generate_memory_hash(memory)

        if memory_hash in self.memories:
            return self.memories[memory_hash]

        self.memories[memory_hash] = memory

        if memory_hash not in self.archived:
            self.archived.append(memory_hash)

        return memory

    # ---------------------------------------------------------
    # SEMANTIC MATCH
    # ---------------------------------------------------------

    def _find_semantic_match(
        self,
        memory: MemoryUnit,
    ):
        """
        Find a semantically similar existing memory while
        preventing merges between conflicting entities.
        """

        try:
            import numpy as np
            import sentence_transformers

            if self._embedding_model is None:
                self._embedding_model = (
                    sentence_transformers.SentenceTransformer(
                        "BAAI/bge-small-en-v1.5"
                    )
                )

            existing_memories = list(self.memories.values())

            if not existing_memories:
                return None

            memory_text = (
                f"{memory.subject} "
                f"{memory.relation} "
                f"{memory.object}"
            )

            existing_texts = [
                f"{existing.subject} "
                f"{existing.relation} "
                f"{existing.object}"
                for existing in existing_memories
            ]

            embeddings = self._embedding_model.encode(
                [memory_text] + existing_texts,
                normalize_embeddings=True,
            )

            candidate_embedding = embeddings[0]
            existing_embeddings = embeddings[1:]

            similarities = np.dot(
                existing_embeddings,
                candidate_embedding,
            )

            best_index = int(np.argmax(similarities))
            best_similarity = float(similarities[best_index])

            existing = existing_memories[best_index]

            print(
                "ACMQ-RAG: semantic store similarity:",
                f"{best_similarity:.4f}"
            )

            if best_similarity < 0.80:
                return None

            candidate_subject = memory.subject.strip().lower()
            existing_subject = existing.subject.strip().lower()

            candidate_relation = memory.relation.strip().lower()
            existing_relation = existing.relation.strip().lower()

            # -------------------------------------------------
            # Prevent false merges between different subjects.
            # Example:
            # John Smith != Jane Smith
            # -------------------------------------------------

            if candidate_subject != existing_subject:

                print(
                    "ACMQ-RAG: semantic match rejected:",
                    "subject conflict ->",
                    f"{existing.subject} != {memory.subject}"
                )

                return None

            # -------------------------------------------------
            # Same subject + sufficiently similar memory
            # -------------------------------------------------

            print(
                "ACMQ-RAG: semantic match accepted:",
                f"{existing.subject} | {existing.relation} | {existing.object}"
            )

            return existing

        except Exception as exc:

            print(
                "ACMQ-RAG: semantic store matching failed:",
                exc
            )

            return None

    # ---------------------------------------------------------
    # SUBJECT + RELATION MATCH
    # ---------------------------------------------------------

    def _find_same_subject_relation(
        self,
        memory: MemoryUnit,
    ):

        subject = memory.subject.strip().lower()
        relation = memory.relation.strip().lower()

        for existing in self.memories.values():

            if (
                existing.subject.strip().lower()
                == subject
                and
                existing.relation.strip().lower()
                == relation
            ):

                return existing

        return None

    # ---------------------------------------------------------
    # REMOVE CLASSIFICATION
    # ---------------------------------------------------------

    def _remove_from_collections(
        self,
        memory_hash: str,
    ):

        collections = [
            self.accepted,
            self.merged,
            self.updated,
            self.archived,
            self.discarded,
        ]

        for collection in collections:

            if memory_hash in collection:
                collection.remove(memory_hash)

    # ---------------------------------------------------------
    # ADD BATCH
    # ---------------------------------------------------------

    def add_batch(
        self,
        memories: List[MemoryUnit],
    ):

        results = []

        for memory in memories:

            result = self.add(memory)

            results.append(result)

        return results

    # ---------------------------------------------------------
    # GET
    # ---------------------------------------------------------

    def get(
        self,
        memory_hash: str,
    ) -> Optional[MemoryUnit]:

        return self.memories.get(memory_hash)

    # ---------------------------------------------------------
    # GET ALL
    # ---------------------------------------------------------

    def get_all(self) -> List[MemoryUnit]:

        return list(self.memories.values())

    # ---------------------------------------------------------
    # ACCEPTED
    # ---------------------------------------------------------

    def get_accepted(self):

        return [
            self.memories[h]
            for h in self.accepted
            if h in self.memories
        ]

    # ---------------------------------------------------------
    # MERGED
    # ---------------------------------------------------------

    def get_merged(self):

        return [
            self.memories[h]
            for h in self.merged
            if h in self.memories
        ]

    # ---------------------------------------------------------
    # UPDATED
    # ---------------------------------------------------------

    def get_updated(self):

        return [
            self.memories[h]
            for h in self.updated
            if h in self.memories
        ]

    # ---------------------------------------------------------
    # ARCHIVED
    # ---------------------------------------------------------

    def get_archived(self):

        return [
            self.memories[h]
            for h in self.archived
            if h in self.memories
        ]

    # ---------------------------------------------------------
    # DISCARDED
    # ---------------------------------------------------------

    def get_discarded(self):

        return [
            self.memories[h]
            for h in self.discarded
            if h in self.memories
        ]

    # ---------------------------------------------------------
    # GRAPH MEMORIES
    # ---------------------------------------------------------

    def get_graph_memories(self):

        graph_memories = []

        graph_memories.extend(
            self.get_accepted()
        )

        graph_memories.extend(
            self.get_updated()
        )

        graph_memories.extend(
            self.get_merged()
        )

        return graph_memories

    # ---------------------------------------------------------
    # CONTAINS
    # ---------------------------------------------------------

    def contains(
        self,
        memory: MemoryUnit,
    ) -> bool:

        memory_hash = generate_memory_hash(memory)

        return memory_hash in self.memories

    # ---------------------------------------------------------
    # CLEAR
    # ---------------------------------------------------------

    def clear(self):

        self.memories.clear()

        self.accepted.clear()
        self.merged.clear()
        self.updated.clear()
        self.archived.clear()
        self.discarded.clear()

        self.decision_history.clear()

        self.merge_history.clear()
        self.update_history.clear()
        self.archive_history.clear()
        self.discard_history.clear()
        self.accept_history.clear()

        self._embedding_model = None


    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    def statistics(self):

        counts = {
            "total_memories": len(self.memories),
            "accepted": 0,
            "merged": 0,
            "updated": 0,
            "archived": 0,
            "discarded": 0,
        }

        for memory in self.memories.values():

            decision = memory.decision

            if decision == MemoryDecision.ACCEPT:
                counts["accepted"] += 1

            elif decision == MemoryDecision.MERGE:
                counts["merged"] += 1

            elif decision == MemoryDecision.UPDATE:
                counts["updated"] += 1

            elif decision == MemoryDecision.ARCHIVE:
                counts["archived"] += 1

            elif decision == MemoryDecision.DISCARD:
                counts["discarded"] += 1

        counts["decision_history"] = len(
            self.decision_history
        )

        counts["merge_events"] = len(
            self.merge_history
        )

        counts["update_events"] = len(
            self.update_history
        )

        counts["archive_events"] = len(
            self.archive_history
        )

        counts["discard_events"] = len(
            self.discard_history
        )

        counts["accept_events"] = len(
            self.accept_history
        )

        return counts

    # ---------------------------------------------------------
    # PRINT STATISTICS
    # ---------------------------------------------------------

    def print_statistics(self):

        stats = self.statistics()

        print()
        print("=" * 50)
        print("ACMQ-RAG Memory Store")
        print("=" * 50)

        for key, value in stats.items():
            print(f"{key:20}: {value}")

        print("=" * 50)
