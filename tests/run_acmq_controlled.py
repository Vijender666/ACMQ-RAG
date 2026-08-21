"""
ACMQ-RAG Controlled Decision Experiment

Runs the controlled dataset through the actual
MemoryPipeline and compares expected vs actual decisions.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hipporag.memory.memory_pipeline import MemoryPipeline
from hipporag.memory.memory_types import MemoryDecision
from acmq_controlled_dataset import get_controlled_cases


def run_case(case):
    """
    Run one controlled case through the actual ACMQ-RAG pipeline.

    Triples are processed sequentially so that duplicate
    and conflicting-memory behavior is evaluated against
    previously stored memories.
    """

    # Controlled graph context for deterministic ACMQ-RAG experiments.
    # High-importance entities receive strong graph statistics.
    # Random/low-importance entities receive weak graph statistics.
    graph_stats = {}

    if case["name"] == "high_quality_fact":
        from hipporag.utils.misc_utils import compute_mdhash_id

        for entity in ["Steve Jobs", "Apple"]:
            key = str(
                compute_mdhash_id(
                    content=entity,
                    prefix="entity-",
                )
            ).lower()

            graph_stats[key] = {
                "degree": 1.0,
                "pagerank": 1.0,
            }

    elif case["name"] == "updated_knowledge":
        from hipporag.utils.misc_utils import compute_mdhash_id

        for entity in ["TechCorp", "John Smith", "Jane Smith"]:
            key = str(
                compute_mdhash_id(
                    content=entity,
                    prefix="entity-",
                )
            ).lower()

            graph_stats[key] = {
                "degree": 1.0,
                "pagerank": 1.0,
            }

    elif case["name"] == "low_importance":
        from hipporag.utils.misc_utils import compute_mdhash_id

        for entity in ["Random Person", "blue pens"]:
            key = str(
                compute_mdhash_id(
                    content=entity,
                    prefix="entity-",
                )
            ).lower()

            graph_stats[key] = {
                "degree": 0.05,
                "pagerank": 0.05,
            }

    pipeline = MemoryPipeline(
        graph_stats=graph_stats
    )

    actual_decisions = []

    for triple in case["triples"]:

        memory_units = pipeline.triples_to_memory_units(
            [triple],
            source_name="controlled_experiment",
        )

        memory_units = pipeline.evaluate(memory_units)
        memory_units = pipeline.decide(memory_units)

        for memory in memory_units:

            actual_decisions.append(
                memory.decision.value.upper()
            )

        pipeline.store(memory_units)

    return actual_decisions


def main():

    cases = get_controlled_cases()

    results = []

    print()
    print("=" * 80)
    print("ACMQ-RAG CONTROLLED DECISION EXPERIMENT")
    print("=" * 80)

    for case in cases:

        expected = case["expected"].upper()

        actual_decisions = run_case(case)

        # For a multi-triple case, the final decision is
        # the decision of the final candidate.
        actual = (
            actual_decisions[-1]
            if actual_decisions
            else "NONE"
        )

        passed = actual == expected

        results.append(passed)

        print()
        print(f"Case     : {case['name']}")
        print(f"Expected : {expected}")
        print(f"Actual   : {actual}")
        print(
            "Status   : "
            + ("PASS" if passed else "FAIL")
        )

        if len(actual_decisions) > 1:
            print(
                f"Decisions: {' -> '.join(actual_decisions)}"
            )

    passed_count = sum(results)
    total_count = len(results)

    print()
    print("=" * 80)
    print(
        f"RESULT: {passed_count}/{total_count} controlled cases passed"
    )
    print("=" * 80)

    if passed_count != total_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
