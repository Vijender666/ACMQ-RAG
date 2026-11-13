import argparse
import csv
import json
import os
import random
import re
import string
import time
from collections import Counter
from pathlib import Path

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import Chunk


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "reproduce" / "dataset"
OUTPUT_DIR = ROOT / "experiments" / "acmq" / "results"


def load_data():
    with open(DATA_DIR / "hotpotqa.json", encoding="utf-8") as f:
        questions = json.load(f)

    with open(DATA_DIR / "hotpotqa_corpus.json", encoding="utf-8") as f:
        corpus = json.load(f)

    return questions, corpus


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(
        ch for ch in text if ch not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, gold):
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction, gold):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def build_corpus(corpus):
    documents = []

    for row in corpus:
        title = str(row["title"]).strip()
        text = str(row["text"]).strip()

        if not title or not text:
            continue

        content = f"Wikipedia Title: {title}\n{text}"

        documents.append(
            Chunk(
                content=content,
                source_id=str(row.get("idx", title)),
                metadata={
                    "title": title,
                    "idx": row.get("idx"),
                },
            )
        )

    return documents


def make_config(save_dir, args):
    return BaseConfig(
        dataset="hotpotqa",
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_model,
        retrieval_top_k=args.top_k,
        qa_top_k=args.qa_top_k,
        save_dir=str(save_dir),
        force_index_from_scratch=args.force_reindex,
        openie_mode="online",
        save_openie=True,
        temperature=0,
    )


def get_retrieval_items(result):
    # Handle APIs that return (results, aggregate_metrics).
    if isinstance(result, tuple):
        result = result[0]

    if isinstance(result, list):
        return result

    raise TypeError(
        f"Unexpected retrieve() result type: {type(result).__name__}"
    )


def get_result_metadata(solution):
    metadata = getattr(solution, "doc_metadata", None)

    if metadata is None and isinstance(solution, dict):
        metadata = solution.get("doc_metadata", [])

    return metadata or []


def get_result_docs(solution):
    docs = getattr(solution, "docs", None)

    if docs is None and isinstance(solution, dict):
        docs = solution.get("docs", [])

    return docs or []


def get_prediction(solution):
    if isinstance(solution, dict):
        return str(
            solution.get("answer")
            or solution.get("predicted_answer")
            or solution.get("response")
            or ""
        )
    return str(getattr(solution, "answer", "") or "")

def calculate_recall(metadata, gold_titles, k):
    retrieved_titles = [
        str(item["title"])
        for item in metadata[:k]
        if isinstance(item, dict) and item.get("title")
    ]
    gold = set(gold_titles)
    if not gold:
        return 0.0
    return len(set(retrieved_titles) & gold) / len(gold)

def run_experiment(name, docs, questions, args):
    save_dir = OUTPUT_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = make_config(save_dir, args)

    print(f"\n{'=' * 65}")
    print(f"Running: {name}")
    print(f"ACMQ enabled: {args.acmq}")
    print(f"Corpus documents: {len(docs)}")
    print(f"Evaluation questions: {len(questions)}")
    print(f"Output directory: {save_dir}")
    print("=" * 65)

    hipporag = HippoRAG(
        global_config=config,
        save_dir=str(save_dir),
    )

    hipporag.enable_acmq = args.acmq

    start = time.perf_counter()
    hipporag.index(docs)
    indexing_seconds = time.perf_counter() - start

    queries = [row["question"] for row in questions]

    start = time.perf_counter()
    retrieved = get_retrieval_items(hipporag.retrieve(queries))
    retrieval_seconds = time.perf_counter() - start

    if len(retrieved) != len(questions):
        raise RuntimeError(
            f"Expected {len(questions)} retrieval results, "
            f"received {len(retrieved)}."
        )

    rows = []

    for question, solution in zip(questions, retrieved):
        gold_titles = list(
            dict.fromkeys(
                str(fact[0])
                for fact in question.get("supporting_facts", [])
                if len(fact) > 0
            )
        )

        metadata = get_result_metadata(solution)
        docs_returned = get_result_docs(solution)

        rows.append({
            "system": name,
            "question_id": question.get("_id", ""),
            "question": question["question"],
            "gold_answer": str(question["answer"]),
            "prediction": get_prediction(solution),
            "gold_supporting_titles": json.dumps(gold_titles),
            "retrieved_titles": json.dumps([
                item.get("title", "")
                for item in metadata
                if isinstance(item, dict)
            ]),
            "retrieved_documents": len(docs_returned),
            "recall_at_1": calculate_recall(metadata, gold_titles, 1),
            "recall_at_5": calculate_recall(metadata, gold_titles, 5),
            "recall_at_10": calculate_recall(metadata, gold_titles, 10),
        })

    # Generate answers using the retrieved QuerySolution objects.
    # This avoids retrieving the same questions a second time.
    start = time.perf_counter()
    qa_output = hipporag.qa(retrieved)
    qa_seconds = time.perf_counter() - start

    if not isinstance(qa_output, tuple) or len(qa_output) < 2:
        raise RuntimeError(
            "Unexpected qa() output. Inspect its returned structure "
            "before calculating answer metrics."
        )

    solved_queries = qa_output[0]

    for row, solution in zip(rows, solved_queries):
        row["prediction"] = get_prediction(solution)

        # Some implementations store the answer in the returned metadata
        # instead of on the QuerySolution.
        if not row["prediction"]:
            row["prediction"] = ""

        row["exact_match"] = exact_match(
            row["prediction"], row["gold_answer"]
        )
        row["f1"] = token_f1(
            row["prediction"], row["gold_answer"]
        )

    summary = {
        "system": name,
        "acmq_enabled": args.acmq,
        "questions_evaluated": len(rows),
        "corpus_documents": len(docs),
        "indexing_seconds": round(indexing_seconds, 3),
        "retrieval_seconds_total": round(retrieval_seconds, 3),
        "qa_seconds_total": round(qa_seconds, 3),
        "recall_at_1": round(
            sum(row["recall_at_1"] for row in rows) / len(rows), 4
        ),
        "recall_at_5": round(
            sum(row["recall_at_5"] for row in rows) / len(rows), 4
        ),
        "recall_at_10": round(
            sum(row["recall_at_10"] for row in rows) / len(rows), 4
        ),
        "exact_match": round(
            sum(row["exact_match"] for row in rows) / len(rows), 4
        ),
        "f1": round(
            sum(row["f1"] for row in rows) / len(rows), 4
        ),
    }

    result_file = OUTPUT_DIR / f"{name}_questions.csv"

    with open(result_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"Question-level results: {result_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compare HippoRAG and ACMQ-RAG on HotpotQA."
    )

    parser.add_argument(
        "--questions", type=int, default=20,
        help="Number of HotpotQA questions for the pilot."
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of documents retrieved per question."
    )
    parser.add_argument(
        "--qa-top-k", type=int, default=5,
        help="Number of retrieved documents passed to answer generation."
    )
    parser.add_argument(
        "--llm-name", default="qwen2.5:7b"
    )
    parser.add_argument(
        "--llm-base-url", default="http://127.0.0.1:11434/v1"
    )
    parser.add_argument(
        "--embedding-model",
        default="Transformers/BAAI/bge-small-en-v1.5"
    )
    parser.add_argument(
        "--force-reindex", action="store_true",
        help="Force a fresh index; this can be expensive."
    )
    parser.add_argument(
        "--system", choices=["baseline", "acmq", "both"], default="both"
    )

    args = parser.parse_args()

    if args.questions < 1:
        parser.error("--questions must be at least 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    questions, corpus = load_data()
    questions = questions[:args.questions]
    docs = build_corpus(corpus)

    print(f"Loaded {len(questions)} questions and {len(docs)} documents.")
    print("NOTE: This pilot uses the first N questions but the full corpus.")
    print("Indexing the full corpus may take considerable time.")

    summaries = []

    if args.system in ("baseline", "both"):
        args.acmq = False
        summaries.append(
            run_experiment("hotpotqa_baseline", docs, questions, args)
        )

    if args.system in ("acmq", "both"):
        args.acmq = True
        summaries.append(
            run_experiment("hotpotqa_acmq", docs, questions, args)
        )

    summary_file = OUTPUT_DIR / "hotpotqa_summary.csv"

    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summaries[0].keys())
        )
        writer.writeheader()
        writer.writerows(summaries)

    print("\nFINAL COMPARISON")
    for summary in summaries:
        print(
            f"{summary['system']}: "
            f"EM={summary['exact_match']:.4f}, "
            f"F1={summary['f1']:.4f}, "
            f"R@5={summary['recall_at_5']:.4f}"
        )

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
