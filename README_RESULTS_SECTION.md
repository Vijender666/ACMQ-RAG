## Experimental Outputs

A small controlled pilot (9 documents, 3 queries, Top-K=5, Qwen2.5:7b) is documented in [`results/RESULTS.md`](results/RESULTS.md), with selected console screenshots in [`results/screenshots/`](results/screenshots/).

In this pilot, Recall@1/2/5 and QA Exact Match were unchanged versus Original HippoRAG. QA F1 changed from 0.3827 to 0.4048 (+0.0221). ACMQ-RAG indexing took longer in this run (9.834 s vs 0.028 s). These are preliminary pilot results, not HotpotQA benchmark results or evidence of general improvement.
