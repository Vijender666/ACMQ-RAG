# ACMQ-RAG Experimental Outputs

## Important context

These screenshots document a **small controlled pilot run**, not the HotpotQA benchmark.
The run shown used 9 documents, 3 queries, Top-K = 5, and the Qwen2.5:7b model.
Do not present these values as full-dataset or HotpotQA results.

## Retrieval and QA comparison

| Metric | Original HippoRAG | ACMQ-RAG | Difference |
|---|---:|---:|---:|
| Recall@1 | 0.6111 | 0.6111 | 0.0000 |
| Recall@2 | 0.8889 | 0.8889 | 0.0000 |
| Recall@5 | 1.0000 | 1.0000 | 0.0000 |
| QA Exact Match | 0.3333 | 0.3333 | 0.0000 |
| QA F1 | 0.3827 | 0.4048 | +0.0221 |

## Runtime reported by the pilot

| Measurement | Original HippoRAG | ACMQ-RAG |
|---|---:|---:|
| Indexing time | 0.028 s | 9.834 s |
| Retrieval time | 120.637 s | 115.887 s |

## Graph and memory statistics shown

The comparison summary reports:
- Original graph: 34 nodes, 71 edges.
- ACMQ graph: 32 nodes, 63 edges.
- Memory store: 16 stored memories (15 active, 1 archived).
- Memory decisions: 11 accepted, 4 updated, 1 merged, 1 archived, 0 discarded.

The quality-score screenshot displays six dimensions: novelty, consistency, reliability, importance, utility, and freshness.

## Interpretation

In this pilot, Recall@1, Recall@2, Recall@5, and Exact Match were unchanged.
QA F1 increased from 0.3827 to 0.4048 (+0.0221).
The reported indexing time was higher for ACMQ-RAG, while retrieval time was slightly lower.

Because this is a three-query, nine-document pilot, the observations are preliminary and are not sufficient to establish general benchmark improvements.
Run larger controlled experiments (including the planned HotpotQA evaluation) before making broader performance claims.

## Screenshots

- `screenshots/01_graph_and_memory_statistics.png`
- `screenshots/02_retrieval_and_timing_comparison.png`
- `screenshots/03_qa_metrics_comparison.png`
- `screenshots/04_memory_decision_summary.png`
- `screenshots/05_memory_quality_scores.png`
- `screenshots/06_selected_knowledge_triples.png`
- `screenshots/07_acmq_retrieved_results.png`
