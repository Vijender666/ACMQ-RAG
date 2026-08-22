"""
Adaptive Memory Decision Engine (MDE)

ACMQ-RAG
"""

from typing import List

from .memory_types import MemoryDecision
from .memory_types import MemoryUnit


class MemoryDecisionEngine:
    """
    Adaptive Memory Decision Engine.

    Decision hierarchy:

    1. Exact duplicate       -> MERGE
    2. Same subject/relation
       stronger evidence    -> UPDATE
       weaker evidence      -> ARCHIVE
    3. Semantic duplicate   -> MERGE
    4. High quality         -> ACCEPT
    5. Medium quality       -> ARCHIVE
    6. Low quality          -> DISCARD
    """

    def __init__(
        self,
        accept_threshold: float = 0.85,
        archive_threshold: float = 0.45,
        semantic_merge_threshold: float = 0.80,
        semantic_update_threshold: float = 0.60,
        critical_reliability_threshold: float = 0.30,
    ):

        self.accept_threshold = accept_threshold
        self.archive_threshold = archive_threshold
        self.semantic_merge_threshold = semantic_merge_threshold
        self.semantic_update_threshold = semantic_update_threshold
        self.critical_reliability_threshold = (
            critical_reliability_threshold
        )

        self._embedding_model = None

    # ---------------------------------------------------------
    # SEMANTIC SIMILARITY
    # ---------------------------------------------------------

    def semantic_similarity(
        self,
        memory: MemoryUnit,
        existing: MemoryUnit,
    ) -> float:

        try:

            import numpy as np
            import sentence_transformers

            if self._embedding_model is None:

                self._embedding_model = (
                    sentence_transformers.SentenceTransformer(
                        "BAAI/bge-small-en-v1.5"
                    )
                )

            memory_text = (
                f"{memory.subject} "
                f"{memory.relation} "
                f"{memory.object}"
            )

            existing_text = (
                f"{existing.subject} "
                f"{existing.relation} "
                f"{existing.object}"
            )

            embeddings = self._embedding_model.encode(
                [
                    memory_text,
                    existing_text,
                ],
                normalize_embeddings=True,
            )

            similarity = float(
                np.dot(
                    embeddings[0],
                    embeddings[1],
                )
            )

            return max(
                0.0,
                min(1.0, similarity),
            )

        except Exception as exc:

            print(
                "ACMQ-RAG: semantic similarity failed:",
                exc,
            )

            return 0.0

    # ---------------------------------------------------------
    # EXACT DUPLICATE
    # ---------------------------------------------------------

    def is_exact_duplicate(
        self,
        memory: MemoryUnit,
        existing: MemoryUnit,
    ) -> bool:

        return (
            memory.subject.strip().lower()
            == existing.subject.strip().lower()
            and
            memory.relation.strip().lower()
            == existing.relation.strip().lower()
            and
            memory.object.strip().lower()
            == existing.object.strip().lower()
        )

    # ---------------------------------------------------------
    # SAME SUBJECT + RELATION
    # ---------------------------------------------------------

    def is_same_relation(
        self,
        memory: MemoryUnit,
        existing: MemoryUnit,
    ) -> bool:

        return (
            memory.subject.strip().lower()
            == existing.subject.strip().lower()
            and
            memory.relation.strip().lower()
            == existing.relation.strip().lower()
        )

    # ---------------------------------------------------------
    # MAIN DECISION
    # ---------------------------------------------------------

    def process(
        self,
        memory: MemoryUnit,
        existing_memories: List[MemoryUnit],
    ) -> MemoryUnit:

        # =====================================================
        # STEP 1: EXACT DUPLICATE
        # =====================================================
        #
        # Exact duplicate = same subject + relation + object.
        # Stronger evidence -> UPDATE.
        # Equal/weaker evidence -> MERGE.
        # =====================================================

        for existing in existing_memories:

            if self.is_exact_duplicate(
                memory,
                existing,
            ):

                new_score = memory.quality.overall_score
                old_score = existing.quality.overall_score

                memory.matched_memory_hash = (
                    self._memory_hash(existing)
                )

                memory.set_decision(
                    MemoryDecision.MERGE
                )

                memory.decision_reason = (
                    "Exact duplicate of existing memory"
                )

                print(
                    "ACMQ-RAG DECISION: MERGE |",
                    memory.as_text(),
                    f"| new={new_score:.3f}",
                    f"| old={old_score:.3f}",
                )

                return memory

        # =====================================================
        # STEP 2: SAME SUBJECT + RELATION
        # =====================================================

        for existing in existing_memories:

            if self.is_same_relation(
                memory,
                existing,
            ):

                new_score = memory.quality.overall_score
                old_score = existing.quality.overall_score

                memory.matched_memory_hash = (
                    self._memory_hash(existing)
                )

                # A conflicting fact for the same subject/relation
                # represents knowledge evolution. UPDATE is allowed
                # when the candidate has sufficient quality and is
                # meaningfully novel, even if its absolute score is
                # slightly below the previous memory.
                #
                # This prevents a newer valid fact from being archived
                # merely because the older fact has accumulated a
                # higher historical quality score.

                if new_score >= self.archive_threshold:

                    memory.set_decision(
                        MemoryDecision.UPDATE
                    )

                    memory.decision_reason = (
                        "Same subject/relation with "
                        "sufficient quality and novel evidence"
                    )

                    print(
                        "ACMQ-RAG DECISION: UPDATE |",
                        memory.as_text(),
                        f"| new={new_score:.3f}",
                        f"| novelty={memory.quality.novelty:.3f}",
                    )

                else:

                    memory.set_decision(
                        MemoryDecision.ARCHIVE
                    )

                    memory.decision_reason = (
                        "Conflicting fact has weaker "
                        "or equal evidence"
                    )

                    print(
                        "ACMQ-RAG DECISION: ARCHIVE |",
                        memory.as_text(),
                        f"| new={new_score:.3f}",
                        f"<= old={old_score:.3f}",
                    )

                return memory

        # =====================================================
        # STEP 3: SEMANTIC SEARCH
        # =====================================================

        best_similarity = 0.0
        best_existing = None

        for existing in existing_memories:

            similarity = self.semantic_similarity(
                memory,
                existing,
            )

            print(
                "ACMQ-RAG SEMANTIC CHECK:",
                memory.as_text(),
                "<->",
                existing.as_text(),
                f"| similarity={similarity:.4f}",
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_existing = existing

        # =====================================================
        # STEP 4: SEMANTIC RELATION DECISION
        # =====================================================

        if best_existing is not None:

            same_subject = (
                memory.subject.strip().lower()
                == best_existing.subject.strip().lower()
            )

            same_object = (
                memory.object.strip().lower()
                == best_existing.object.strip().lower()
            )

            same_relation = (
                memory.relation.strip().lower()
                == best_existing.relation.strip().lower()
            )

            # -------------------------------------------------
            # SEMANTIC UPDATE
            # -------------------------------------------------
            #
            # Same subject + sufficiently high semantic
            # similarity means this is related/evolving
            # knowledge, even when relation wording differs.
            #
            # Example:
            #
            # Tim Cook --[leads]--> Apple
            # Tim Cook --[is the chief executive of]--> Apple
            #
            # These should produce UPDATE.
            # -------------------------------------------------

            if (
                same_subject
                and best_similarity >= self.semantic_update_threshold
            ):

                memory.set_decision(
                    MemoryDecision.UPDATE
                )

                memory.matched_memory_hash = (
                    self._memory_hash(best_existing)
                )

                memory.decision_reason = (
                    "Semantically similar memory with "
                    "matching subject and evolving evidence"
                )

                print(
                    "ACMQ-RAG DECISION: UPDATE |",
                    memory.as_text(),
                    f"| similarity={best_similarity:.3f}",
                )

                return memory

            # -------------------------------------------------
            # SEMANTIC MERGE
            # -------------------------------------------------
            #
            # Same subject + object + relation with very high
            # similarity represents a semantic duplicate.
            # -------------------------------------------------

            if (
                same_subject
                and same_object
                and same_relation
                and best_similarity >= self.semantic_merge_threshold
            ):

                memory.set_decision(
                    MemoryDecision.MERGE
                )

                memory.matched_memory_hash = (
                    self._memory_hash(best_existing)
                )

                memory.decision_reason = (
                    "Semantic duplicate with matching "
                    "subject, relation, and object"
                )

                print(
                    "ACMQ-RAG DECISION: MERGE |",
                    memory.as_text(),
                    f"| similarity={best_similarity:.3f}",
                )

                return memory

        # =====================================================
        # STEP 5: CONTENT QUALITY
        # =====================================================

        if memory.quality.content_quality <= 0.0:

            memory.set_decision(
                MemoryDecision.DISCARD
            )

            memory.decision_reason = (
                "Invalid or zero content quality"
            )

            print(
                "ACMQ-RAG DECISION: DISCARD |",
                memory.as_text(),
            )

            return memory

        # =====================================================
        # STEP 6: QUALITY DECISION
        # =====================================================
        #
        # ACMQ-RAG uses both:
        #
        # 1. Overall multi-factor quality score
        # 2. Critical quality safeguards
        #
        # This prevents a memory with critically poor
        # reliability, consistency, or content quality
        # from being accepted solely because other factors
        # compensate for it.
        # =====================================================

        score = memory.quality.overall_score

        reliability = memory.quality.source_reliability
        consistency = memory.quality.semantic_consistency
        content_quality = memory.quality.content_quality

        # -----------------------------------------------------
        # CRITICAL QUALITY SAFEGUARD
        # -----------------------------------------------------

        # Only apply dimension-level safeguards when the memory
        # has actually been evaluated by the quality model.
        #
        # This preserves compatibility with manually constructed
        # MemoryUnits that provide an overall quality score directly.

        if memory.quality.evaluated:

            if reliability <= self.critical_reliability_threshold:

                memory.set_decision(
                    MemoryDecision.DISCARD
                )

                memory.decision_reason = (
                    "Critically low source reliability"
                )

                print(
                    "ACMQ-RAG DECISION: DISCARD |",
                    memory.as_text(),
                    f"| reliability={reliability:.3f}",
                    f"| threshold="
                    f"{self.critical_reliability_threshold:.3f}",
                )

                return memory

            if consistency <= 0.0:

                memory.set_decision(
                    MemoryDecision.DISCARD
                )

                memory.decision_reason = (
                    "Zero semantic consistency"
                )

                print(
                    "ACMQ-RAG DECISION: DISCARD |",
                    memory.as_text(),
                    f"| consistency={consistency:.3f}",
                )

                return memory

        if content_quality <= 0.0:

            memory.set_decision(
                MemoryDecision.DISCARD
            )

            memory.decision_reason = (
                "Invalid or zero content quality"
            )

            print(
                "ACMQ-RAG DECISION: DISCARD |",
                memory.as_text(),
            )

            return memory

        # -----------------------------------------------------
        # OVERALL QUALITY DECISION
        # -----------------------------------------------------

        if score >= self.accept_threshold:

            memory.set_decision(
                MemoryDecision.ACCEPT
            )

            memory.decision_reason = (
                "Novel high-quality memory"
            )

            print(
                "ACMQ-RAG DECISION: ACCEPT |",
                memory.as_text(),
                f"| score={score:.3f}",
            )

        elif score >= self.archive_threshold:

            memory.set_decision(
                MemoryDecision.ARCHIVE
            )

            memory.decision_reason = (
                "Novel medium-quality memory"
            )

            print(
                "ACMQ-RAG DECISION: ARCHIVE |",
                memory.as_text(),
                f"| score={score:.3f}",
            )

        else:

            memory.set_decision(
                MemoryDecision.DISCARD
            )

            memory.decision_reason = (
                "Novel low-quality memory"
            )

            print(
                "ACMQ-RAG DECISION: DISCARD |",
                memory.as_text(),
                f"| score={score:.3f}",
            )

        return memory

    # ---------------------------------------------------------
    # BATCH PROCESSING
    # ---------------------------------------------------------

    def process_batch(
        self,
        memories: List[MemoryUnit],
        existing_memories: List[MemoryUnit] = None,
    ) -> List[MemoryUnit]:

        processed = []

        if existing_memories is None:
            existing_memories = []

        # IMPORTANT:
        # Work on a copy so candidate decisions do not
        # accidentally mutate the persistent store.

        context = list(existing_memories)

        for memory in memories:

            memory = self.process(
                memory,
                context,
            )

            processed.append(memory)

            # Newly accepted/updated memories become
            # context for later candidates.

            if memory.decision in (
                MemoryDecision.ACCEPT,
                MemoryDecision.UPDATE,
            ):

                context.append(memory)

        return processed

    # ---------------------------------------------------------
    # HASH HELPER
    # ---------------------------------------------------------

    @staticmethod
    def _memory_hash(memory: MemoryUnit) -> str:

        import hashlib

        value = (
            f"{memory.subject.strip().lower()}|"
            f"{memory.relation.strip().lower()}|"
            f"{memory.object.strip().lower()}"
        )

        return hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()
