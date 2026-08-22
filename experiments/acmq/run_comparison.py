"""
ACMQ-RAG vs Baseline Comparative Experiment

Compares:
    1. Rule-based baseline
    2. ACMQ-RAG adaptive memory decision engine

The experiment focuses on cases where quality-aware
memory construction should provide behavior beyond
simple duplicate/update rules.
"""

from src.hipporag.memory.memory_types import MemoryUnit
from src.hipporag.memory.memory_quality_model import MemoryQualityModel
from src.hipporag.memory.memory_decision_engine import MemoryDecisionEngine


def make_memory(
    subject,
    relation,
    obj,
    reliability=1.0,
    timestamp=None,
):
    memory = MemoryUnit(
        subject=subject,
        relation=relation,
        object=obj,
    )

    memory.metadata.reliability = reliability
    memory.metadata.timestamp = timestamp

    return memory


def baseline_decision(memory, existing):
    """
    Simple rule-based baseline.

    No quality scoring.
    No graph importance.
    No retrieval utility.
    No temporal freshness.
    """

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

    if any(v in invalid for v in values):
        return "DISCARD"

    for old in existing:

        if (
            old.subject.strip().lower()
            == memory.subject.strip().lower()
            and
            old.relation.strip().lower()
            == memory.relation.strip().lower()
            and
            old.object.strip().lower()
            == memory.object.strip().lower()
        ):
            return "MERGE"

    for old in existing:

        if (
            old.subject.strip().lower()
            == memory.subject.strip().lower()
            and
            old.relation.strip().lower()
            == memory.relation.strip().lower()
        ):
            return "UPDATE"

    return "ACCEPT"


def run_case(
    name,
    candidate,
    existing,
    graph_stats,
):
    print()
    print("=" * 70)
    print(f"CASE: {name}")
    print("=" * 70)

    baseline = baseline_decision(
        candidate,
        existing,
    )

    print(
        f"Baseline decision : {baseline}"
    )

    quality_model = MemoryQualityModel(
        graph_stats=graph_stats,
    )

    engine = MemoryDecisionEngine()

    quality_model.evaluate(
        candidate,
        existing,
    )

    result = engine.process(
        candidate,
        existing,
    )

    print(
        f"ACMQ-RAG decision  : "
        f"{result.decision.value.upper()}"
    )

    print(
        f"Quality score      : "
        f"{result.quality.overall_score:.4f}"
    )

    print(
        f"Importance         : "
        f"{result.quality.graph_importance:.4f}"
    )

    print(
        f"Retrieval utility  : "
        f"{result.quality.retrieval_utility:.4f}"
    )

    print(
        f"Freshness          : "
        f"{result.quality.temporal_freshness:.4f}"
    )

    print(
        f"Reason             : "
        f"{result.decision_reason}"
    )

    return baseline, result.decision.value.upper()


def main():

    print("=" * 70)
    print("ACMQ-RAG vs BASELINE COMPARISON")
    print("=" * 70)

    cases = []

    # ---------------------------------------------------------
    # CASE 1
    # High-quality important fact
    # ---------------------------------------------------------

    cases.append(
        (
            "high_quality_fact",
            make_memory(
                "Steve Jobs",
                "founded",
                "Apple",
                reliability=1.0,
            ),
            [],
            {
                "entity-7f76ff8615d64e788ea5e9633def1625": {
                    "degree": 1.0,
                    "pagerank": 1.0,
                },
                "entity-9f6290f4436e5a2351f12e03b6433c3c": {
                    "degree": 1.0,
                    "pagerank": 1.0,
                },
            },
        )
    )

    # ---------------------------------------------------------
    # CASE 2
    # Exact duplicate
    # ---------------------------------------------------------

    old = make_memory(
        "Steve Jobs",
        "founded",
        "Apple",
    )

    cases.append(
        (
            "exact_duplicate",
            make_memory(
                "Steve Jobs",
                "founded",
                "Apple",
            ),
            [old],
            {
                "entity-7f76ff8615d64e788ea5e9633def1625": {
                    "degree": 1.0,
                    "pagerank": 1.0,
                },
                "entity-9f6290f4436e5a2351f12e03b6433c3c": {
                    "degree": 1.0,
                    "pagerank": 1.0,
                },
            },
        )
    )

    # ---------------------------------------------------------
    # CASE 3
    # Low-quality / low-reliability memory
    # ---------------------------------------------------------

    cases.append(
        (
            "low_reliability_fact",
            make_memory(
                "Random Source",
                "claims",
                "Unverified Fact",
                reliability=0.10,
            ),
            [],
            {},
        )
    )

    # ---------------------------------------------------------
    # CASE 4
    # Low graph importance
    # ---------------------------------------------------------

    cases.append(
        (
            "low_graph_importance",
            make_memory(
                "Random Person",
                "likes",
                "blue pens",
                reliability=1.0,
            ),
            [],
            {
                "entity-a4112b61193f9eec73cfeb84f3bbf5e9": {
                    "degree": 0.15,
                    "pagerank": 0.10,
                },
                "entity-dfbe75863903d767b793b3c1118b7af1": {
                    "degree": 0.10,
                    "pagerank": 0.05,
                },
            },
        )
    )

    # ---------------------------------------------------------
    # RUN
    # ---------------------------------------------------------

    baseline_results = []
    acmq_results = []

    for case in cases:

        baseline, acmq = run_case(
            *case
        )

        baseline_results.append(
            baseline
        )

        acmq_results.append(
            acmq
        )

    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for i, case in enumerate(cases):

        print(
            f"{case[0]:25} "
            f"Baseline={baseline_results[i]:8} "
            f"ACMQ-RAG={acmq_results[i]:8}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
