"""
Controlled ACMQ-RAG memory decision experiment.

Tests:
    ACCEPT
    MERGE
    UPDATE
    ARCHIVE
    DISCARD

The experiment supplies controlled graph statistics so that
memory quality is evaluated under reproducible conditions.
"""

from tests.acmq_controlled_dataset import get_controlled_cases

from hipporag.memory.memory_types import MemoryUnit
from hipporag.memory.memory_quality_model import MemoryQualityModel
from hipporag.memory.memory_decision_engine import MemoryDecisionEngine
from hipporag.memory.memory_store import MemoryStore


def make_memory(triple):
    subject, relation, obj = triple

    return MemoryUnit(
        subject=subject,
        relation=relation,
        object=obj,
    )


def controlled_graph_stats(case_name):
    """
    Provide deterministic graph statistics for each experiment case.

    The values are intentionally controlled so the experiment can
    exercise all ACMQ-RAG decision branches.
    """

    from hipporag.utils.misc_utils import compute_mdhash_id

    def entity_key(text):
        return str(
            compute_mdhash_id(
                content=text,
                prefix="entity-",
            )
        ).lower()

    if case_name == "high_quality_fact":
        values = {
            "Steve Jobs": (1.0, 1.0),
            "Apple": (1.0, 1.0),
        }

    elif case_name == "exact_duplicate":
        values = {
            "Steve Jobs": (1.0, 1.0),
            "Apple": (1.0, 1.0),
        }

    elif case_name == "updated_knowledge":
        values = {
            "TechCorp": (1.0, 1.0),
            "John Smith": (0.6, 0.6),
            "Jane Smith": (0.9, 0.9),
        }

    elif case_name == "low_importance":
        values = {
            "Random Person": (0.15, 0.10),
            "blue pens": (0.10, 0.05),
        }

    elif case_name == "low_quality":
        values = {
            "unknown": (0.0, 0.0),
        }

    else:
        values = {}

    return {
        entity_key(name): {
            "degree": degree,
            "pagerank": pagerank,
        }
        for name, (degree, pagerank) in values.items()
    }


def run_case(case):
    print()
    print("=" * 70)
    print(f"CASE: {case['name']}")
    print("=" * 70)

    memories = [
        make_memory(triple)
        for triple in case["triples"]
    ]

    graph_stats = controlled_graph_stats(case["name"])

    quality_model = MemoryQualityModel(
        graph_stats=graph_stats
    )

    decision_engine = MemoryDecisionEngine()
    store = MemoryStore()

    print(f"Expected decision: {case['expected']}")
    print()

    print("Controlled Graph Statistics:")
    print("-" * 70)

    for key, stats in graph_stats.items():
        print(
            f"{key} "
            f"degree={stats['degree']:.3f} "
            f"pagerank={stats['pagerank']:.3f}"
        )

    print()

    print("Memory Quality:")
    print("-" * 70)

    evaluated = quality_model.evaluate_batch(
        memories,
        store.get_all(),
    )

    for memory in evaluated:

        q = memory.quality

        print(
            f"{memory.subject} | "
            f"{memory.relation} | "
            f"{memory.object}"
        )

        print(
            f"  novelty={q.novelty:.3f} "
            f"consistency={q.semantic_consistency:.3f} "
            f"reliability={q.source_reliability:.3f} "
            f"content={q.content_quality:.3f} "
            f"importance={q.graph_importance:.3f} "
            f"utility={q.retrieval_utility:.3f} "
            f"freshness={q.temporal_freshness:.3f} "
            f"overall={q.overall_score:.3f}"
        )

    print()
    print("Decisions:")
    print("-" * 70)

    decided = decision_engine.process_batch(
        evaluated,
        store.get_all(),
    )

    for memory in decided:

        print(
            f"{memory.subject} | "
            f"{memory.relation} | "
            f"{memory.object}"
        )

        print(
            f"  decision={memory.decision}"
        )

    return decided


def main():

    cases = get_controlled_cases()

    print()
    print("=" * 70)
    print("ACMQ-RAG CONTROLLED MEMORY DECISION EXPERIMENT")
    print("=" * 70)

    all_results = []

    for case in cases:

        results = run_case(case)

        all_results.append(
            {
                "case": case,
                "results": results,
            }
        )

    print()
    print("=" * 70)
    print("CONTROLLED EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
