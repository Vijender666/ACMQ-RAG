import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hipporag.memory.memory_types import (
    MemoryUnit,
    MemoryDecision,
)
from hipporag.memory.memory_quality_model import MemoryQualityModel
from hipporag.memory.memory_decision_engine import MemoryDecisionEngine
from hipporag.memory.memory_store import MemoryStore
from hipporag.memory.memory_pipeline import MemoryPipeline


# ============================================================
# TEST 1 — QUALITY MODEL
# ============================================================

def test_quality_model_graph_statistics():

    model = MemoryQualityModel()

    subject = "Steve Jobs"
    obj = "Apple"

    memory = MemoryUnit(
        subject=subject,
        relation="founded",
        object=obj,
    )

    # Use hashed entity IDs, matching HippoRAG graph statistics.
    from hipporag.utils.misc_utils import compute_mdhash_id

    subject_key = str(
        compute_mdhash_id(
            content=subject,
            prefix="entity-",
        )
    ).lower()

    object_key = str(
        compute_mdhash_id(
            content=obj,
            prefix="entity-",
        )
    ).lower()

    model.graph_stats = {
        subject_key: {
            "degree": 1.0,
            "pagerank": 0.8,
        },
        object_key: {
            "degree": 0.9,
            "pagerank": 0.4,
        },
    }

    graph_importance = model.compute_graph_importance(memory)

    assert graph_importance == 0.9


# ============================================================
# TEST 2 — DECISION ENGINE
# ============================================================

def test_decision_engine():

    engine = MemoryDecisionEngine()

    memory = MemoryUnit(
        subject="Steve Jobs",
        relation="founded",
        object="Apple",
    )

    memory.quality.overall_score = 0.95
    memory.quality.content_quality = 1.0
    memory.quality.evaluated = True

    result = engine.process(
        memory,
        [],
    )

    assert result.decision == MemoryDecision.ACCEPT


# ============================================================
# TEST 3 — EXACT DUPLICATE
# ============================================================

def test_exact_duplicate_merge():

    engine = MemoryDecisionEngine()

    existing = MemoryUnit(
        subject="Steve Jobs",
        relation="founded",
        object="Apple",
    )

    memory = MemoryUnit(
        subject="Steve Jobs",
        relation="founded",
        object="Apple",
    )

    memory.quality.overall_score = 0.95
    memory.quality.evaluated = True

    result = engine.process(
        memory,
        [existing],
    )

    assert result.decision == MemoryDecision.MERGE


# ============================================================
# TEST 4 — UPDATED KNOWLEDGE
# ============================================================

def test_updated_knowledge():

    engine = MemoryDecisionEngine()

    existing = MemoryUnit(
        subject="Steve Jobs",
        relation="founded",
        object="Apple",
    )

    memory = MemoryUnit(
        subject="Steve Jobs",
        relation="founded",
        object="NeXT",
    )

    memory.quality.overall_score = 0.95
    memory.quality.novelty = 0.30
    memory.quality.evaluated = True

    result = engine.process(
        memory,
        [existing],
    )

    assert result.decision == MemoryDecision.UPDATE


# ============================================================
# TEST 5 — MEMORY STORE
# ============================================================

def test_memory_store():

    store = MemoryStore()

    memory = MemoryUnit(
        subject="Steve Jobs",
        relation="founded",
        object="Apple",
    )

    memory.decision = MemoryDecision.ACCEPT

    store.add(memory)

    assert len(store.get_all()) == 1
    assert len(store.get_accepted()) == 1
    assert memory in store.get_graph_memories()


# ============================================================
# TEST 6 — DUPLICATE PREVENTION
# ============================================================

def test_memory_store_duplicate_prevention():

    store = MemoryStore()

    memory = MemoryUnit(
        subject="Steve Jobs",
        relation="founded",
        object="Apple",
    )

    memory.decision = MemoryDecision.ACCEPT

    store.add(memory)
    store.add(memory)

    assert len(store.get_all()) == 1


# ============================================================
# TEST 7 — FULL PIPELINE
# ============================================================

def test_full_memory_pipeline():

    pipeline = MemoryPipeline()

    triples = [
        ("Steve Jobs", "founded", "Apple"),
        ("Tim Cook", "leads", "Apple"),
        ("Microsoft", "developed", "Windows"),
    ]

    store = pipeline.process(triples)

    assert len(store.get_all()) == 3

    statistics = store.statistics()

    assert statistics["total_memories"] == 3

    for memory in store.get_all():
        assert memory.quality.evaluated is True
        assert memory.decision is not None


# ============================================================
# TEST 8 — GRAPH MEMORIES
# ============================================================

def test_graph_memories():

    store = MemoryStore()

    accepted = MemoryUnit(
        subject="Steve Jobs",
        relation="founded",
        object="Apple",
    )
    accepted.decision = MemoryDecision.ACCEPT

    archived = MemoryUnit(
        subject="Unknown Person",
        relation="related_to",
        object="Unknown Thing",
    )
    archived.decision = MemoryDecision.ARCHIVE

    store.add(accepted)
    store.add(archived)

    graph_memories = store.get_graph_memories()

    assert accepted in graph_memories
    assert archived not in graph_memories


# ============================================================
# TEST 9 — PIPELINE STATISTICS
# ============================================================

def test_pipeline_statistics():

    pipeline = MemoryPipeline()

    triples = [
        ("Steve Jobs", "founded", "Apple"),
    ]

    pipeline.process(triples)

    stats = pipeline.statistics()

    assert stats["total_memories"] == 1


# ============================================================
# FINAL
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
