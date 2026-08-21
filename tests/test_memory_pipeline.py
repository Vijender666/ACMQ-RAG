"""
Test script for ACMQ-RAG Memory Pipeline
"""
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hipporag.memory import MemoryPipeline


def main():

    pipeline = MemoryPipeline()

    triples = [

        (
            "Steve Jobs",
            "founded",
            "Apple"
        ),

        (
            "Apple",
            "headquartered in",
            "California"
        ),

        (
            "Tim Cook",
            "CEO of",
            "Apple"
        ),

        (
            "Steve Jobs",
            "founded",
            "Apple"
        )

    ]

    graph_memories = pipeline.process(
        triples,
        source_name="Wikipedia",
        chunk_id="chunk_001"
    )

    print("\n========== GRAPH MEMORIES ==========\n")

    graph_memories = graph_memories.get_graph_memories()

    for memory in graph_memories:

        print(memory)

    print()

    pipeline.print_statistics()


if __name__ == "__main__":
    main()