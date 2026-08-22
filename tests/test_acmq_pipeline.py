from hipporag.memory.memory_pipeline import MemoryPipeline
from hipporag.memory.memory_types import MemoryDecision


def test_acmq_pipeline_accepts_high_quality_memory():

    pipeline = MemoryPipeline(
        graph_stats={}
    )

    memories = pipeline.triples_to_memory_units(
        [
            (
                "Steve Jobs",
                "founded",
                "Apple",
            )
        ],
        source_name="test_document",
        chunk_id="chunk-1",
    )

    # Bypass the real embedding-heavy quality model
    # and test the pipeline flow using a controlled score.
    memories[0].quality.novelty = 1.0
    memories[0].quality.semantic_consistency = 1.0
    memories[0].quality.source_reliability = 1.0
    memories[0].quality.content_quality = 1.0
    memories[0].quality.graph_importance = 1.0
    memories[0].quality.retrieval_utility = 1.0
    memories[0].quality.temporal_freshness = 1.0
    memories[0].quality.overall_score = 0.95
    memories[0].quality.evaluated = True

    decided = pipeline.decide(memories)

    pipeline.store(decided)

    stats = pipeline.statistics()

    assert decided[0].decision == MemoryDecision.ACCEPT
    assert stats["total_stored"] == 1
    assert stats["active"] == 1


def test_acmq_pipeline_preserves_chunk_metadata():

    pipeline = MemoryPipeline(
        graph_stats={}
    )

    memories = pipeline.triples_to_memory_units(
        [
            (
                "Tim Cook",
                "leads",
                "Apple",
            )
        ],
        source_name="document-A",
        chunk_id="chunk-42",
    )

    assert len(memories) == 1

    memory = memories[0]

    assert memory.metadata.source_name == "document-A"
    assert memory.metadata.chunk_id == "chunk-42"
    assert memory.subject == "Tim Cook"
    assert memory.relation == "leads"
    assert memory.object == "Apple"


def test_acmq_pipeline_discards_unreliable_memory():

    pipeline = MemoryPipeline(
        graph_stats={}
    )

    memories = pipeline.triples_to_memory_units(
        [
            (
                "Unknown",
                "claims",
                "Unknown Fact",
            )
        ],
        source_name="unreliable-source",
        chunk_id="chunk-3",
        source_reliability=0.10,
    )

    memories[0].quality.novelty = 1.0
    memories[0].quality.semantic_consistency = 1.0
    memories[0].quality.source_reliability = 0.10
    memories[0].quality.content_quality = 1.0
    memories[0].quality.graph_importance = 1.0
    memories[0].quality.retrieval_utility = 1.0
    memories[0].quality.temporal_freshness = 1.0
    memories[0].quality.overall_score = 0.90
    memories[0].quality.evaluated = True

    decided = pipeline.decide(memories)

    pipeline.store(decided)

    stats = pipeline.statistics()

    assert decided[0].decision == MemoryDecision.DISCARD
    assert stats["discarded"] == 1
