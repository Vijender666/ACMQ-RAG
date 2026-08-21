from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.sample_data import DOCS


print("=" * 70)
print("ACMQ-RAG REAL HIPPO-RAG INTEGRATION TEST")
print("=" * 70)

print()
print("Documents:", len(DOCS))

config = BaseConfig(
    llm_name="qwen2.5:7b",
    llm_base_url="http://localhost:11434/v1",
)

print()
print("LLM Configuration:")
print("  Model :", config.llm_name)
print("  URL   :", config.llm_base_url)

rag = HippoRAG(global_config=config)

print()
print("=" * 70)
print("STARTING REAL INDEXING")
print("=" * 70)

rag.index(docs=DOCS)

print()
print("=" * 70)
print("INDEXING COMPLETE")
print("=" * 70)

print()
print("ACMQ-RAG statistics:")

try:
    print(rag.memory_pipeline.statistics())
except Exception as e:
    print("Could not read memory statistics:", e)

print()
print("Graph statistics:")

try:
    print("Graph nodes:", rag.graph.vcount())
    print("Graph edges:", rag.graph.ecount())
except Exception as e:
    print("Could not read graph statistics:", e)

print()
print("ACMQ graph statistics:")

try:
    print("ACMQ nodes:", len(rag.acmq_graph_stats))
except Exception as e:
    print("Could not read ACMQ graph statistics:", e)

print()
print("=" * 70)
print("REAL ACMQ-RAG INTEGRATION TEST COMPLETE")
print("=" * 70)
