"""
ACMQ-RAG vs Original HippoRAG
Baseline Retrieval Experiment
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.sample_data import DOCS, QUERIES, GOLD_DOCS, ANSWERS



TOP_K = 5


def run_qa_evaluation(model, name):
    print()
    print("=" * 80)
    print(f"{name} QA EVALUATION")
    print("=" * 80)

    (
        qa_solutions,
        _,
        _,
        retrieval_metrics,
        qa_metrics,
    ) = model.rag_qa(
        queries=QUERIES,
        gold_docs=GOLD_DOCS,
        gold_answers=ANSWERS,
    )

    print()
    print(f"{name} QA RESULTS")
    print("-" * 80)

    for i, solution in enumerate(qa_solutions):
        print(f"Query {i+1}: {QUERIES[i]}")
        print(f"  Gold     : {ANSWERS[i]}")
        print(f"  Predicted: {solution.answer}")
        print()

    if not isinstance(qa_metrics, dict):
        raise TypeError(
            f"Expected QA metrics as a dictionary, got "
            f"{type(qa_metrics).__name__}: {qa_metrics!r}"
        )

    qa_em = float(qa_metrics["ExactMatch"])
    qa_f1 = float(qa_metrics["F1"])

    print(f"{name} QA Exact Match: {qa_em:.4f}")
    print(f"{name} QA F1        : {qa_f1:.4f}")
    print(f"{name} Retrieval Metrics: {retrieval_metrics}")

    return qa_em, qa_f1


config = BaseConfig(
    llm_name="qwen2.5:7b",
    llm_base_url="http://localhost:11434/v1",
)

print()
print("=" * 80)
print("ACMQ-RAG vs ORIGINAL HIPPO-RAG")
print("BASELINE RETRIEVAL EXPERIMENT")
print("=" * 80)

print()
print("Documents :", len(DOCS))
print("Queries   :", len(QUERIES))
print("Top-K     :", TOP_K)
print("LLM       :", config.llm_name)


# ==============================================================
# ORIGINAL HIPPO-RAG
# ==============================================================

print()
print("=" * 80)
print("ORIGINAL HIPPO-RAG")
print("=" * 80)

baseline = HippoRAG(
    global_config=config
)

start = time.time()

baseline.index(
    docs=DOCS
)

baseline_index_time = time.time() - start

print(
    f"Indexing time: {baseline_index_time:.3f}s"
)

start = time.time()

baseline_results, baseline_metrics = baseline.retrieve(
    queries=QUERIES,
    num_to_retrieve=TOP_K,
    gold_docs=GOLD_DOCS,
)

baseline_retrieval_time = time.time() - start

print(
    f"Retrieval time: {baseline_retrieval_time:.3f}s"
)


print()
print("ORIGINAL RESULTS")
print("-" * 80)

for i, result in enumerate(baseline_results, 1):

    print()
    query = QUERIES[i - 1] if i <= len(QUERIES) else "<missing query>"
    print(f"Query {i}: {query}")

    for rank, doc in enumerate(result.docs, 1):
        print(f"  {rank}. {doc}")

print()
print("Original metrics:")

for metric, value in baseline_metrics.items():
    print(f"  {metric}: {value}")


# ==============================================================
# ACMQ-RAG
# ==============================================================

print()
print("=" * 80)
print("ACMQ-RAG + HIPPO-RAG")
print("=" * 80)

acmq = HippoRAG(
    global_config=config
)
acmq.enable_acmq = True

start = time.time()

acmq.index(
    docs=DOCS
)

acmq_index_time = time.time() - start

print(
    f"Indexing time: {acmq_index_time:.3f}s"
)

start = time.time()

acmq_results, acmq_metrics = acmq.retrieve(
    queries=QUERIES,
    num_to_retrieve=TOP_K,
    gold_docs=GOLD_DOCS,
)

acmq_retrieval_time = time.time() - start


print()
print("ACMQ RESULTS")
print("-" * 80)

for i, result in enumerate(acmq_results, 1):

    print()
    print(f"Query {i}: {QUERIES[i - 1]}")

    for rank, doc in enumerate(result.docs, 1):
        print(f"  {rank}. {doc}")

print()
print("ACMQ metrics:")

for metric, value in acmq_metrics.items():
    print(f"  {metric}: {value}")


# ==============================================================
# COMPARISON
# ==============================================================

print()

# ============================================================
# QA EVALUATION
# ============================================================

original_qa_em, original_qa_f1 = run_qa_evaluation(
    baseline,
    "ORIGINAL HIPPO-RAG"
)

acmq_qa_em, acmq_qa_f1 = run_qa_evaluation(
    acmq,
    "ACMQ-RAG + HIPPO-RAG"
)

print()
print("=" * 80)
print("QA COMPARISON")
print("=" * 80)

print()
print("RAW QA METRICS")
print("-" * 65)
print("Original EM :", original_qa_em)
print("Original F1 :", original_qa_f1)
print("ACMQ EM     :", acmq_qa_em)
print("ACMQ F1     :", acmq_qa_f1)
print()

print("Metric              Original       ACMQ-RAG       Delta")
print("-" * 65)

# Normalize the metric structures returned by rag_qa()
def extract_em(metric):
    if isinstance(metric, dict):
        if "ExactMatch" in metric:
            return metric["ExactMatch"]
        if "EM" in metric:
            return metric["EM"]
        if "exact_match" in metric:
            return metric["exact_match"]
    return float(metric)

def extract_f1(metric):
    if isinstance(metric, dict):
        if "F1" in metric:
            return metric["F1"]
        if "f1" in metric:
            return metric["f1"]
    return float(metric)

original_em = extract_em(original_qa_em)
acmq_em = extract_em(acmq_qa_em)

original_f1 = extract_f1(original_qa_f1)
acmq_f1 = extract_f1(acmq_qa_f1)

print(
    f"QA Exact Match      "
    f"{original_em:.4f}         "
    f"{acmq_em:.4f}         "
    f"{acmq_em - original_em:+.4f}"
)

print(
    f"QA F1               "
    f"{original_f1:.4f}         "
    f"{acmq_f1:.4f}         "
    f"{acmq_f1 - original_f1:+.4f}"
)

print("=" * 80)
print("RETRIEVAL COMPARISON")
print("=" * 80)

print()

print(
    f"{'Metric':<20}"
    f"{'Original':<15}"
    f"{'ACMQ-RAG':<15}"
    f"{'Delta':<15}"
)

print("-" * 65)

metrics = sorted(
    set(baseline_metrics) |
    set(acmq_metrics)
)

for metric in metrics:

    original = baseline_metrics.get(metric, 0.0)
    proposed = acmq_metrics.get(metric, 0.0)
    delta = proposed - original

    print(
        f"{metric:<20}"
        f"{original:<15.4f}"
        f"{proposed:<15.4f}"
        f"{delta:<15.4f}"
    )


# ==============================================================
# TIMING
# ==============================================================

print()
print("=" * 80)
print("TIMING")
print("=" * 80)

print(
    f"Original indexing : {baseline_index_time:.3f}s"
)

print(
    f"ACMQ indexing      : {acmq_index_time:.3f}s"
)

print(
    f"Original retrieval : {baseline_retrieval_time:.3f}s"
)

print(
    f"ACMQ retrieval      : {acmq_retrieval_time:.3f}s"
)


# ==============================================================
# GRAPH
# ==============================================================

print()
print("=" * 80)
print("GRAPH STATISTICS")
print("=" * 80)

try:

    print(
        "Original nodes:",
        baseline.graph.vcount()
    )

    print(
        "Original edges:",
        baseline.graph.ecount()
    )

except Exception as e:

    print(
        "Original graph statistics unavailable:",
        e
    )


try:

    print(
        "ACMQ nodes:",
        acmq.graph.vcount()
    )

    print(
        "ACMQ edges:",
        acmq.graph.ecount()
    )

except Exception as e:

    print(
        "ACMQ graph statistics unavailable:",
        e
    )


# ==============================================================
# MEMORY
# ==============================================================

print()
print("=" * 80)
print("ACMQ MEMORY STATISTICS")
print("=" * 80)

try:

    print(
        acmq.memory_pipeline.statistics()
    )

except Exception as e:

    print(
        "Memory statistics unavailable:",
        e
    )


print()
print("=" * 80)
print("BASELINE EXPERIMENT COMPLETE")
print("=" * 80)
