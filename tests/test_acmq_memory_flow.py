from hipporag.memory.memory_types import (
    MemoryUnit,
    MemoryDecision,
    QualityScore,
)
from hipporag.memory.memory_decision_engine import (
    MemoryDecisionEngine,
)


def make_memory(
    subject,
    relation,
    object_,
    score,
    novelty=1.0,
    reliability=1.0,
    consistency=1.0,
    content_quality=1.0,
):
    memory = MemoryUnit(
        subject=subject,
        relation=relation,
        object=object_,
    )

    memory.quality = QualityScore(
        novelty=novelty,
        semantic_consistency=consistency,
        source_reliability=reliability,
        content_quality=content_quality,
        graph_importance=1.0,
        retrieval_utility=1.0,
        temporal_freshness=1.0,
        overall_score=score,
        evaluated=True,
    )

    return memory


def test_accept_high_quality_memory():
    engine = MemoryDecisionEngine()

    memory = make_memory(
        "Steve Jobs",
        "founded",
        "Apple",
        score=0.90,
    )

    result = engine.process(memory, [])

    assert result.decision == MemoryDecision.ACCEPT


def test_merge_exact_duplicate():
    engine = MemoryDecisionEngine()

    existing = make_memory(
        "Steve Jobs",
        "founded",
        "Apple",
        score=0.90,
    )

    new_memory = make_memory(
        "Steve Jobs",
        "founded",
        "Apple",
        score=0.90,
    )

    result = engine.process(
        new_memory,
        [existing],
    )

    assert result.decision == MemoryDecision.MERGE


def test_update_conflicting_same_relation():
    engine = MemoryDecisionEngine()

    existing = make_memory(
        "Apple",
        "CEO",
        "Tim Cook",
        score=0.80,
    )

    new_memory = make_memory(
        "Apple",
        "CEO",
        "John Smith",
        score=0.70,
    )

    result = engine.process(
        new_memory,
        [existing],
    )

    assert result.decision == MemoryDecision.UPDATE


def test_archive_low_quality_conflicting_memory():
    engine = MemoryDecisionEngine()

    existing = make_memory(
        "Apple",
        "CEO",
        "Tim Cook",
        score=0.90,
    )

    new_memory = make_memory(
        "Apple",
        "CEO",
        "Unknown Person",
        score=0.30,
    )

    result = engine.process(
        new_memory,
        [existing],
    )

    assert result.decision == MemoryDecision.ARCHIVE


def test_discard_critically_unreliable_memory():
    engine = MemoryDecisionEngine()

    memory = make_memory(
        "Random Source",
        "claims",
        "Unknown Fact",
        score=0.90,
        reliability=0.10,
    )

    result = engine.process(
        memory,
        [],
    )

    assert result.decision == MemoryDecision.DISCARD
