import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hipporag.memory.memory_decision_engine import MemoryDecisionEngine
from src.hipporag.memory.memory_types import MemoryUnit


def make_memory(subject, relation, obj):
    return MemoryUnit(
        subject=subject,
        relation=relation,
        object=obj,
    )


def test_exact_duplicate_similarity():
    engine = MemoryDecisionEngine()

    memory_a = make_memory(
        "john smith",
        "is",
        "ceo of techcorp",
    )

    memory_b = make_memory(
        "john smith",
        "is",
        "ceo of techcorp",
    )

    similarity = engine.semantic_similarity(
        memory_a,
        memory_b,
    )

    print(f"\nEXACT DUPLICATE similarity={similarity:.4f}")

    assert similarity >= 0.95


def test_same_subject_relation_similarity():
    engine = MemoryDecisionEngine()

    memory_a = make_memory(
        "john smith",
        "is",
        "ceo of techcorp",
    )

    memory_b = make_memory(
        "john smith",
        "is",
        "ceo of anothercorp",
    )

    similarity = engine.semantic_similarity(
        memory_a,
        memory_b,
    )

    print(
        f"\nSAME SUBJECT + RELATION similarity={similarity:.4f}"
    )

    assert similarity > 0.50


def test_different_subject_similarity():
    engine = MemoryDecisionEngine()

    memory_a = make_memory(
        "john smith",
        "is",
        "ceo of techcorp",
    )

    memory_b = make_memory(
        "jane smith",
        "is",
        "ceo of techcorp",
    )

    similarity = engine.semantic_similarity(
        memory_a,
        memory_b,
    )

    print(
        f"\nDIFFERENT SUBJECT similarity={similarity:.4f}"
    )

    assert similarity < 0.95


def test_unrelated_similarity():
    engine = MemoryDecisionEngine()

    memory_a = make_memory(
        "john smith",
        "is",
        "ceo of techcorp",
    )

    memory_b = make_memory(
        "cinderella",
        "attended",
        "royal ball",
    )

    similarity = engine.semantic_similarity(
        memory_a,
        memory_b,
    )

    print(
        f"\nUNRELATED similarity={similarity:.4f}"
    )

    assert similarity < 0.80
