from hipporag.memory.memory_types import (
    MemoryUnit,
    QualityScore,
)
from hipporag.memory.memory_decision_engine import (
    MemoryDecisionEngine,
)


def make_memory(
    subject,
    relation,
    obj,
    score,
    novelty=1.0,
    content_quality=1.0,
):
    memory = MemoryUnit(
        subject=subject,
        relation=relation,
        object=obj,
    )

    memory.quality = QualityScore(
        novelty=novelty,
        semantic_consistency=1.0,
        source_reliability=1.0,
        content_quality=content_quality,
        graph_importance=1.0,
        retrieval_utility=1.0,
        temporal_freshness=1.0,
        overall_score=score,
    )

    return memory


def test_new_high_quality_memory():
    engine = MemoryDecisionEngine()

    candidate = make_memory(
        "Steve Jobs",
        "founded",
        "Apple",
        0.92,
    )

    result = engine.process(candidate, [])

    assert result.decision.value == "accept"


def test_exact_duplicate():
    engine = MemoryDecisionEngine()

    existing = make_memory(
        "Steve Jobs",
        "founded",
        "Apple",
        0.90,
    )

    candidate = make_memory(
        "Steve Jobs",
        "founded",
        "Apple",
        0.88,
    )

    result = engine.process(
        candidate,
        [existing],
    )

    assert result.decision.value == "merge"


def test_same_subject_relation_update():
    engine = MemoryDecisionEngine()

    existing = make_memory(
        "Apple",
        "CEO",
        "Tim Cook",
        0.90,
    )

    candidate = make_memory(
        "Apple",
        "CEO",
        "John Doe",
        0.75,
    )

    result = engine.process(
        candidate,
        [existing],
    )

    assert result.decision.value == "update"


def test_low_content_quality_discard():
    engine = MemoryDecisionEngine()

    candidate = make_memory(
        "Unknown",
        "unknown",
        "unknown",
        0.20,
        content_quality=0.0,
    )

    result = engine.process(candidate, [])

    assert result.decision.value == "discard"


def test_medium_quality_archive():
    engine = MemoryDecisionEngine()

    candidate = make_memory(
        "Microsoft",
        "created",
        "Windows",
        0.60,
    )

    result = engine.process(candidate, [])

    assert result.decision.value == "archive"


def test_semantic_relation_update():
    engine = MemoryDecisionEngine(
        semantic_merge_threshold=0.80,
        semantic_update_threshold=0.60,
    )

    existing = make_memory(
        "Tim Cook",
        "leads",
        "Apple",
        0.80,
    )

    candidate = make_memory(
        "Tim Cook",
        "is the chief executive of",
        "Apple",
        0.70,
    )

    result = engine.process(
        candidate,
        [existing],
    )

    assert result.decision.value == "update"


def test_semantic_match_records_existing_memory():
    engine = MemoryDecisionEngine(
        semantic_merge_threshold=0.80,
        semantic_update_threshold=0.60,
    )

    existing = make_memory(
        "Tim Cook",
        "leads",
        "Apple",
        0.80,
    )

    candidate = make_memory(
        "Tim Cook",
        "is the chief executive of",
        "Apple",
        0.70,
    )

    result = engine.process(
        candidate,
        [existing],
    )

    assert result.decision.value == "update"
    assert result.matched_memory_hash is not None
