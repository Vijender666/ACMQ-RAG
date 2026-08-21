import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hipporag.memory.memory_decision_engine import MemoryDecisionEngine
from src.hipporag.memory.memory_types import MemoryUnit


def make_memory(subject, relation, obj):
    return MemoryUnit(
        subject=subject,
        relation=relation,
        object=obj,
    )


def main():

    engine = MemoryDecisionEngine()

    cases = [
        (
            "EXACT DUPLICATE",
            ("john smith", "is", "ceo of techcorp"),
            ("john smith", "is", "ceo of techcorp"),
        ),
        (
            "SAME SUBJECT + RELATION",
            ("john smith", "is", "ceo of techcorp"),
            ("john smith", "is", "ceo of anothercorp"),
        ),
        (
            "DIFFERENT SUBJECT",
            ("john smith", "is", "ceo of techcorp"),
            ("jane smith", "is", "ceo of techcorp"),
        ),
        (
            "UNRELATED",
            ("john smith", "is", "ceo of techcorp"),
            ("cinderella", "attended", "royal ball"),
        ),
    ]

    print("\n========== ACMQ-RAG SEMANTIC SIMILARITY ==========\n")

    for name, first, second in cases:

        memory_a = make_memory(*first)
        memory_b = make_memory(*second)

        similarity = engine.semantic_similarity(
            memory_a,
            memory_b,
        )

        print(f"{name:<25} : {similarity:.4f}")

    print("\n==================================================\n")


if __name__ == "__main__":
    main()
