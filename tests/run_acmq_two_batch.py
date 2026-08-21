from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


print("=" * 70)
print("ACMQ-RAG TWO-BATCH MEMORY EVOLUTION TEST")
print("=" * 70)

config = BaseConfig(
    llm_name="qwen2.5:7b",
    llm_base_url="http://localhost:11434/v1",
)

rag = HippoRAG(global_config=config)

# ==============================================================
# BATCH 1
# ==============================================================

docs_batch_1 = [
    "TechCorp is a technology company. John Smith is the CEO of TechCorp."
]

print()
print("=" * 70)
print("BATCH 1: INITIAL KNOWLEDGE")
print("=" * 70)

rag.index(docs=docs_batch_1)

print()
print("Memory statistics after Batch 1:")

try:
    print(rag.memory_pipeline.statistics())
except Exception as e:
    print("Statistics error:", e)


# ==============================================================
# BATCH 2
# ==============================================================

docs_batch_2 = [
    "According to an official TechCorp announcement, Jane Smith is now the CEO of TechCorp. The announcement confirms that Jane Smith has replaced John Smith as CEO."
]

print()
print("=" * 70)
print("BATCH 2: UPDATED KNOWLEDGE")
print("=" * 70)

rag.index(docs=docs_batch_2)

print()
print("Memory statistics after Batch 2:")

try:
    print(rag.memory_pipeline.statistics())
except Exception as e:
    print("Statistics error:", e)


# ==============================================================
# FINAL MEMORY STORE
# ==============================================================

print()
print("=" * 70)
print("FINAL ACMQ-RAG MEMORY STORE")
print("=" * 70)

try:
    memories = rag.memory_pipeline.memory_store.get_all()

    print("Total memories:", len(memories))

    for memory in memories:
        print()
        print(
            f"{memory.subject} | "
            f"{memory.relation} | "
            f"{memory.object}"
        )

        print(
            "Decision:",
            memory.decision
        )

        print(
            "Quality:",
            f"{memory.quality.overall_score:.3f}"
        )

except Exception as e:
    print("Memory store error:", e)


print()
print("=" * 70)
print("TWO-BATCH TEST COMPLETE")
print("=" * 70)
