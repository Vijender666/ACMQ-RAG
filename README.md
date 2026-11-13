# ACMQ-RAG
### Adaptive Memory Construction for Graph-Based Retrieval-Augmented Generation

ACMQ-RAG is a research project that explores adaptive memory construction to improve Graph-based Retrieval-Augmented Generation (GraphRAG).

The framework introduces a Context-Aware Memory Quality Model (CMQM) to evaluate candidate knowledge units before incorporating them into the memory construction process.

The goal is to improve the quality and usefulness of stored knowledge while reducing unnecessary or low-quality information in graph-based retrieval systems.

---

## 1. Research Motivation

Traditional Retrieval-Augmented Generation (RAG) systems retrieve relevant information from external knowledge sources to generate answers.

Graph-based RAG systems extend this approach by representing relationships between entities and knowledge units in a graph.

However, memory construction can introduce challenges:

- Redundant knowledge units.
- Irrelevant information.
- Semantically inconsistent information.
- Unnecessary graph complexity.
- Difficulty maintaining useful knowledge.

ACMQ-RAG investigates an adaptive approach to address these challenges through quality-aware memory construction.

## 2. Proposed Approach

The proposed framework extends the HippoRAG architecture with an adaptive memory construction mechanism.

Its central component is the **Context-Aware Memory Quality Model (CMQM)**.

CMQM evaluates candidate knowledge units using six quality dimensions, including:

- Novelty
- Semantic consistency
- Other context-aware quality dimensions

The quality assessment helps determine which knowledge units should be considered during memory construction.

### Key Objectives

1. Improve the quality of constructed memory.
2. Reduce redundant and unnecessary knowledge.
3. Maintain semantically consistent information.
4. Improve retrieval effectiveness.
5. Evaluate the impact of adaptive memory construction on answer generation.

## 3. System Architecture

```text
              Input Documents
                     |
                     v
              Document Processing
                     |
                     v
              Knowledge Extraction
                     |
                     v
              Candidate Knowledge Units
                     |
                     v
        Context-Aware Memory Quality Model
                     |
                     v
            Quality Assessment
                     |
                     v
          Adaptive Memory Construction
                     |
                     v
              Knowledge Graph
                     |
                     v
              Graph Retrieval
                     |
                     v
              LLM Answer Generation
                     |
                     v
                 Final Answer
```

*Conceptual workflow of the proposed ACMQ-RAG framework.*

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| HippoRAG | Graph-based RAG foundation |
| Ollama | Local LLM execution |
| Qwen 2.5 7B | Language model |
| BGE-small-en-v1.5 | Text embeddings |
| OpenAI-compatible API | LLM integration interface |
| HotpotQA | Benchmark evaluation |
| Git & GitHub | Version control |

## 5. Datasets

### HotpotQA

HotpotQA is used as a benchmark for evaluating multi-hop question answering.

The benchmark setup in this repository uses:

- `hotpotqa.json` – Questions and reference answers.
- `hotpotqa_corpus.json` – Document corpus.

The current evaluation script supports comparing the baseline and ACMQ-enabled configurations.

## 6. Evaluation Metrics

The project uses the following evaluation measures:

| Metric | Description |
|---|---|
| Recall@K | Measures retrieval of relevant supporting documents |
| Exact Match (EM) | Measures exact agreement with reference answers |
| F1 Score | Measures token-level overlap between predicted and reference answers |
| Indexing Time | Measures the time required to construct the index |
| Retrieval Time | Measures retrieval execution time |

Actual benchmark results will be added after completing the controlled experiments.

## 7. Project Structure

```text
ACMQ-RAG/
│
├── src/
│   └── hipporag/
│       └── HippoRAG.py
│
├── tests/
│   └── run_acmq_baseline.py
│
├── experiments/
│   └── acmq/
│       └── run_acmq_hotpotqa.py
│
├── reproduce/
│   └── dataset/
│       ├── hotpotqa.json
│       └── hotpotqa_corpus.json
│
├── .gitignore
└── README.md
```

*Illustrative structure showing the principal project files. Additional upstream HippoRAG files and local experiment outputs are omitted.*

## 8. Installation and Setup

### Prerequisites

- Python
- Conda or another Python environment manager
- Git
- Ollama
- Qwen 2.5 7B model

### Clone the repository

```bash
git clone -b acmq-experiments https://github.com/Vijender666/ACMQ-RAG.git

cd ACMQ-RAG
```

### Create and activate the environment

Use your project's existing environment configuration and dependency installation instructions.

### Install and run Ollama

Install Ollama from:

https://ollama.com/

Download the model:

```bash
ollama pull qwen2.5:7b
```

Start the Ollama server:

```bash
ollama serve
```

## 9. Running Experiments

Set the API compatibility key for the local Ollama endpoint:

```bash
export OPENAI_API_KEY=ollama
```

### Baseline Experiment

Run the original baseline test:

```bash
python tests/run_acmq_baseline.py
```

### HotpotQA Baseline Evaluation

```bash
python experiments/acmq/run_acmq_hotpotqa.py \
  --system baseline \
  --questions 20 \
  --top-k 10 \
  --qa-top-k 5
```

### ACMQ-RAG Evaluation

```bash
python experiments/acmq/run_acmq_hotpotqa.py \
  --system acmq \
  --questions 20 \
  --top-k 10 \
  --qa-top-k 5
```

Ensure the benchmark datasets and required dependencies are available before running the experiments.

## 10. Research Contribution

The main research direction of ACMQ-RAG is adaptive, quality-aware memory construction for graph-based retrieval systems.

The framework investigates how evaluating candidate knowledge units before memory construction can influence:

- Memory quality.
- Knowledge redundancy.
- Graph construction.
- Retrieval effectiveness.
- Question-answering performance.

The effectiveness of these contributions must be established through controlled experimental evaluation.

## 11. Future Work

- Evaluate on larger benchmark datasets.
- Investigate memory quality assessment strategies.
- Improve adaptive knowledge selection.
- Compare against additional GraphRAG baselines.
- Conduct ablation studies on individual CMQM dimensions.
- Evaluate indexing overhead and retrieval efficiency.

## 12. Repository

**GitHub:** https://github.com/Vijender666/ACMQ-RAG

**Research Project:** ACMQ-RAG – Adaptive Memory Construction for Graph-Based Retrieval-Augmented Generation.

---

## Author

**Vijender Reddy Talakanti**

Computer Science and Engineering – Data Science

Research Interests:
- Artificial Intelligence
- Large Language Models
- Retrieval-Augmented Generation
- Knowledge Graphs
- Adaptive Memory Systems

---EOF
