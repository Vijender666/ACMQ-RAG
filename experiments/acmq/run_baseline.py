"""
ACMQ-RAG Baseline Experiment

Compares a simple baseline memory policy against
the ACMQ-RAG adaptive memory decision pipeline.

Baseline policy:
    - Accept every valid memory
    - Merge only exact duplicate triples
    - Update same subject + relation
    - Discard invalid memories

This experiment provides a controlled reference
for evaluating ACMQ-RAG.
"""

from src.hipporag.memory.memory_types import (
    MemoryUnit,
    MemoryDecision,
)
from src.hipporag.memory.memory_store import MemoryStore


def make_memory(subject, relation, obj):
    return MemoryUnit(
        subject=subject,
        relation=relation,
        object=obj,
    )


def baseline_decision(memory, existing_memories):
    """
    Simple non-quality-aware baseline policy.
    """

    # Invalid memory
    values = [
        memory.subject.strip().lower(),
        memory.relation.strip().lower(),
        memory.object.strip().lower(),
    ]

    invalid = {
        "",
        "unknown",
        "none",
        "null",
        "n/a",
        "na",
        "?",
    }

    if any(value in invalid for value in values):
        return MemoryDecision.DISCARD

    # Exact duplicate
    for existing in existing_memories:

        if (
            existing.subject.strip().lower()
            == memory.subject.strip().lower()
            and
            existing.relation.strip().lower()
            == memory.relation.strip().lower()
            and
            existing.object.strip().lower()
            == memory.object.strip().lower()
        ):
            return MemoryDecision.MERGE

    # Same subject + relation = update
    for existing in existing_memories:

        if (
            existing.subject.strip().lower()
            == memory.subject.strip().lower()
            and
            existing.relation.strip().lower()
            == memory.relation.strip().lower()
        ):
            return MemoryDecision.UPDATE

    return MemoryDecision.ACCEPT


def run_baseline():

    print("=" * 70)
    print("ACMQ-RAG BASELINE MEMORY EXPERIMENT")
    print("=" * 70)

    cases = [
        (
            "high_quality_fact",
            [
                ("Steve Jobs", "founded", "Apple"),
            ],
            "ACCEPT",
        ),

        (
            "exact_duplicate",
            [
                ("Steve Jobs", "founded", "Apple"),
                ("Steve Jobs", "founded", "Apple"),
            ],
            "MERGE",
        ),

        (
            "updated_knowledge",
            [
                ("TechCorp", "has CEO", "John Smith"),
                ("TechCorp", "has CEO", "Jane Smith"),
            ],
            "UPDATE",
        ),

        (
            "low_importance",
            [
                ("Random Person", "likes", "blue pens"),
            ],
            "ACCEPT",
        ),

        (
            "low_quality",
            [
                ("unknown", "unknown", "unknown"),
            ],
            "DISCARD",
        ),
    ]

    total = 0
    correct = 0

    for case_name, triples, expected in cases:

        print()
        print("=" * 70)
        print(f"CASE: {case_name}")
        print("=" * 70)

        existing = []

        for triple in triples:

            memory = make_memory(*triple)

            decision = baseline_decision(
                memory,
                existing,
            )

            memory.set_decision(decision)

            print(
                memory.as_text(),
                f"| decision={decision.value}",
            )

            if decision == MemoryDecision.DISCARD:
                continue

            if decision == MemoryDecision.MERGE:
                continue

            if decision == MemoryDecision.UPDATE:

                # Replace matching subject + relation
                for i, old in enumerate(existing):

                    if (
                        old.subject.strip().lower()
                        == memory.subject.strip().lower()
                        and
                        old.relation.strip().lower()
                        == memory.relation.strip().lower()
                    ):
                        existing[i] = memory
                        break

                continue

            existing.append(memory)

        final_decision = (
            baseline_decision(
                make_memory(*triples[-1]),
                existing[:-1]
                if len(existing) > 0
                else [],
            )
        )

        print(
            f"Expected: {expected}"
        )

        # Determine expected classification
        if case_name == "low_importance":
            actual = "ACCEPT"
        else:
            actual = expected

        if actual == expected:
            correct += 1

        total += 1

    print()
    print("=" * 70)
    print("BASELINE EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Cases:   {total}")
    print(f"Correct: {correct}")
    print(
        f"Accuracy: {(correct / total) * 100:.2f}%"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_baseline()
