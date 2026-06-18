"""
RAG evaluation: retrieval metrics + LLM-judged RAG quality metrics.

Usage
-----
    python evaluate.py                          # eval all confluence Qs in eval_dataset.jsonl
    python evaluate.py --dataset questions.jsonl --source-type confluence
    python evaluate.py --retrieval-only         # no answer generation / LLM judge
    python evaluate.py --limit 5                  # first N questions only
    python evaluate.py --output results/eval.json

eval_dataset.jsonl is the confluence-only subset of questions.jsonl (64 questions).
Rebuild it after questions.jsonl changes:
    python evaluate.py --build-dataset

Dataset format (JSONL, one object per line)
-------------------------------------------
{
  "question_id": "qst_0013",
  "question": "...",
  "expected_doc_ids": ["dsid_abc123..."],     # dsid prefix in filename
  "gold_answer": "...",
  "answer_facts": ["fact 1", "fact 2", ...]    # optional, for fact recall
}

Metrics
-------
Retrieval (no LLM cost):
  - doc_hit@1          expected doc is the #1 expanded source
  - doc_hit@k          expected doc appears in top_sources
  - doc_in_context     expected doc appears in chunks sent to LLM
  - chunk_mrr          mean reciprocal rank of first matching chunk
  - retrieval_latency_ms

RAG quality (LLM judge via OpenRouter — needs OPENROUTER_API_KEY):
  - contextual_recall      can the gold answer be supported by retrieved context?
  - contextual_precision   are retrieved chunks relevant and not noisy?
  - contextual_relevancy   is retrieved context relevant to the question?
  - faithfulness           is the generated answer grounded in context?
  - answer_correctness     does the answer match the gold answer?
  - fact_recall            fraction of answer_facts found in generated answer
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Import pipeline after dotenv so API keys are available.
from pipline import (  # noqa: E402
    FINAL_K,
    OPENROUTER_API_KEY,
    OPENROUTER_MODELS,
    generate_answer,
    ensure_loaded,
    retrieval_context_string,
    search,
)

DEFAULT_DATASET = "./eval_dataset.jsonl"
QUESTIONS_SOURCE = "./questions.jsonl"
DEFAULT_OUTPUT = "./eval_results.json"

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", OPENROUTER_MODELS[0] if OPENROUTER_MODELS else "")


def load_dataset(path, source_type=None, confluence_only=False):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_no}: invalid JSON ({e})")
                continue
            types = obj.get("source_types") or []
            if confluence_only:
                if types != ["confluence"]:
                    continue
            elif source_type and source_type not in types:
                continue
            rows.append(obj)
    return rows


def build_eval_dataset(
    src=QUESTIONS_SOURCE,
    dst=DEFAULT_DATASET,
    source_type="confluence",
    confluence_only=True,
):
    """Write eval subset from questions.jsonl.

    Default: only questions with source_types == ["confluence"] (index has
    confluence docs only). Loose filter (confluence in source_types) pulls in
    multi-source questions whose gold docs may live in jira/slack — unfair eval.
    """
    rows = load_dataset(
        src,
        source_type=None if confluence_only else source_type,
        confluence_only=confluence_only,
    )
    with open(dst, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    label = "confluence-only" if confluence_only else source_type
    print(f"Wrote {len(rows)} {label} question(s) from {src} -> {dst}")
    return len(rows)


def source_matches_doc_id(source_filename, doc_id):
    """True if filename contains the expected dsid (e.g. dsid_f1b208...)."""
    if not source_filename or not doc_id:
        return False
    base = os.path.basename(source_filename)
    return doc_id in base


def retrieval_metrics(expected_doc_ids, debug, context_chunks):
    """Objective retrieval scores for one question."""
    expected = expected_doc_ids or []
    top_sources = debug.get("top_sources") or []

    doc_hit_at_1 = False
    if expected and top_sources:
        doc_hit_at_1 = any(
            source_matches_doc_id(top_sources[0], d) for d in expected
        )

    doc_hit_at_k = False
    if expected and top_sources:
        doc_hit_at_k = any(
            any(source_matches_doc_id(src, d) for d in expected)
            for src in top_sources
        )

    context_sources = [c["metadata"].get("source", "") for c in context_chunks]
    doc_in_context = False
    if expected:
        doc_in_context = any(
            any(source_matches_doc_id(src, d) for d in expected)
            for src in context_sources
        )

    # MRR over expanded rerank list (all chunks from selected sources).
    mrr = 0.0
    expanded = debug.get("expanded_ranked") or context_chunks
    if expected:
        for rank, chunk in enumerate(expanded, 1):
            src = chunk["metadata"].get("source", "")
            if any(source_matches_doc_id(src, d) for d in expected):
                mrr = 1.0 / rank
                break

    primary_rank = None
    for i, src in enumerate(top_sources):
        if any(source_matches_doc_id(src, d) for d in expected):
            primary_rank = i + 1
            break

    return {
        "doc_hit@1": doc_hit_at_1,
        "doc_hit@k": doc_hit_at_k,
        "doc_in_context": doc_in_context,
        "chunk_mrr": mrr,
        "expected_primary_rank": primary_rank,
        "top_sources": top_sources,
        "context_chunk_count": len(context_chunks),
    }


def fact_recall(answer_facts, generated_answer):
    """Fraction of gold facts whose key tokens appear in the answer."""
    if not answer_facts or not generated_answer:
        return None
    ans = generated_answer.lower()

    def fact_hit(fact):
        tokens = [t for t in re.findall(r"[a-z0-9]+", fact.lower()) if len(t) > 3]
        if not tokens:
            return False
        hits = sum(1 for t in tokens if t in ans)
        return hits / len(tokens) >= 0.5

    hits = sum(1 for f in answer_facts if fact_hit(f))
    return hits / len(answer_facts)


def _parse_judge_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def llm_judge_scores(question, context, gold_answer, generated_answer, api_key):
    """
    One judge call returning DeepEval-style RAG metrics (0.0–1.0 each).

    Definitions align with common RAG eval frameworks:
    - Contextual Recall: all information in gold_answer attributable to context?
    - Contextual Precision: retrieved context is precise (relevant, low noise)?
    - Contextual Relevancy: context is relevant to the question?
    - Faithfulness: generated answer only uses context (no hallucination)?
    - Answer Correctness: generated answer matches gold_answer semantically?
    """
    if not api_key or not JUDGE_MODEL:
        return None

    prompt = f"""You are an expert RAG evaluator. Score each metric from 0.0 to 1.0.

Definitions:
- contextual_recall: Fraction of claims in the GOLD ANSWER that can be attributed to the RETRIEVED CONTEXT (1.0 = fully supported).
- contextual_precision: Fraction of the RETRIEVED CONTEXT that is useful for answering the question (1.0 = no irrelevant noise).
- contextual_relevancy: How relevant the RETRIEVED CONTEXT is to the QUESTION (1.0 = highly relevant).
- faithfulness: Fraction of claims in the GENERATED ANSWER supported by RETRIEVED CONTEXT (1.0 = fully grounded).
- answer_correctness: Semantic overlap between GENERATED ANSWER and GOLD ANSWER (1.0 = equivalent).

Return ONLY valid JSON with these keys (numbers 0.0-1.0) and a short "notes" string:
{{"contextual_recall": 0.0, "contextual_precision": 0.0, "contextual_relevancy": 0.0, "faithfulness": 0.0, "answer_correctness": 0.0, "notes": "..."}}

QUESTION:
{question}

RETRIEVED CONTEXT:
{context[:12000]}

GOLD ANSWER:
{gold_answer}

GENERATED ANSWER:
{generated_answer or "(no answer generated)"}
"""

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
    except Exception as e:
        return {"error": str(e)}

    msg = resp.choices[0].message if resp.choices else None
    content = (msg.content if msg else None) or ""
    parsed = _parse_judge_json(content)
    if parsed is None:
        return {"error": "judge returned non-JSON", "raw": content[:500]}
    parsed["judge_model"] = JUDGE_MODEL
    return parsed


def evaluate_one(row, index, chunks, model, reranker, bm25, *,
                 retrieval_only=False, generate=True):
    qid = row.get("question_id", "?")
    question = row["question"]
    expected_doc_ids = row.get("expected_doc_ids") or []
    gold_answer = row.get("gold_answer", "")
    answer_facts = row.get("answer_facts") or []

    t0 = time.perf_counter()
    context_chunks, debug = search(index, model, reranker, bm25, question, chunks)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    ret = retrieval_metrics(expected_doc_ids, debug, context_chunks)
    ret["retrieval_latency_ms"] = round(retrieval_ms, 1)

    context_text = retrieval_context_string(context_chunks)
    generated_answer = None
    gen_model = None
    judge = None
    facts = None

    if not retrieval_only and generate and OPENROUTER_API_KEY:
        t1 = time.perf_counter()
        generated_answer, gen_model = generate_answer(
            question, context_chunks, OPENROUTER_API_KEY
        )
        gen_ms = (time.perf_counter() - t1) * 1000
        ret["generation_latency_ms"] = round(gen_ms, 1)
        ret["generation_model"] = gen_model

        facts = fact_recall(answer_facts, generated_answer or "")
        judge = llm_judge_scores(
            question, context_text, gold_answer,
            generated_answer or "", OPENROUTER_API_KEY,
        )
        if judge and "error" not in judge:
            judge["fact_recall"] = facts

    return {
        "question_id": qid,
        "question": question,
        "expected_doc_ids": expected_doc_ids,
        "retrieval": ret,
        "generated_answer": generated_answer,
        "gold_answer": gold_answer,
        "judge": judge,
        "fact_recall": facts,
    }


def aggregate(results):
    """Mean scores across all evaluated questions."""
    if not results:
        return {}

    def avg_bool(key):
        vals = [r["retrieval"][key] for r in results if key in r["retrieval"]]
        return round(mean(1.0 if v else 0.0 for v in vals), 4) if vals else None

    def avg_num(path):
        vals = []
        for r in results:
            obj = r
            for part in path:
                obj = (obj or {}).get(part)
            if isinstance(obj, (int, float)):
                vals.append(float(obj))
        return round(mean(vals), 4) if vals else None

    agg = {
        "n": len(results),
        "doc_hit@1": avg_bool("doc_hit@1"),
        "doc_hit@k": avg_bool("doc_hit@k"),
        "doc_in_context": avg_bool("doc_in_context"),
        "mean_chunk_mrr": avg_num(["retrieval", "chunk_mrr"]),
        "mean_retrieval_latency_ms": avg_num(["retrieval", "retrieval_latency_ms"]),
        "mean_contextual_recall": avg_num(["judge", "contextual_recall"]),
        "mean_contextual_precision": avg_num(["judge", "contextual_precision"]),
        "mean_contextual_relevancy": avg_num(["judge", "contextual_relevancy"]),
        "mean_faithfulness": avg_num(["judge", "faithfulness"]),
        "mean_answer_correctness": avg_num(["judge", "answer_correctness"]),
        "mean_fact_recall": avg_num(["fact_recall"]),
    }
    return agg


def print_report(results, aggregate_scores):
    print("\n" + "=" * 72)
    print("RAG EVALUATION REPORT")
    print("=" * 72)

    for r in results:
        print(f"\n--- {r['question_id']} ---")
        ret = r["retrieval"]
        print(
            f"  Retrieval: hit@1={ret['doc_hit@1']}  hit@k={ret['doc_hit@k']}  "
            f"in_context={ret['doc_in_context']}  MRR={ret['chunk_mrr']:.3f}  "
            f"latency={ret['retrieval_latency_ms']}ms"
        )
        if ret.get("top_sources"):
            print(f"  Sources: {ret['top_sources'][0][:70]}...")
        j = r.get("judge")
        if j and "error" not in j:
            print(
                f"  Contextual recall={j.get('contextual_recall')}  "
                f"precision={j.get('contextual_precision')}  "
                f"relevancy={j.get('contextual_relevancy')}"
            )
            print(
                f"  Faithfulness={j.get('faithfulness')}  "
                f"answer_correctness={j.get('answer_correctness')}  "
                f"fact_recall={r.get('fact_recall')}"
            )
        elif j and "error" in j:
            print(f"  Judge error: {j['error']}")
        if r.get("generated_answer"):
            print(f"  Answer preview: {r['generated_answer'][:120]}...")

    print("\n" + "-" * 72)
    print("AGGREGATE (mean over dataset)")
    print("-" * 72)
    labels = [
        ("doc_hit@1", "Doc Hit@1"),
        ("doc_hit@k", "Doc Hit@k"),
        ("doc_in_context", "Doc in LLM context"),
        ("mean_chunk_mrr", "Chunk MRR"),
        ("mean_retrieval_latency_ms", "Retrieval latency (ms)"),
        ("mean_contextual_recall", "Contextual Recall"),
        ("mean_contextual_precision", "Contextual Precision"),
        ("mean_contextual_relevancy", "Contextual Relevancy"),
        ("mean_faithfulness", "Faithfulness"),
        ("mean_answer_correctness", "Answer Correctness"),
        ("mean_fact_recall", "Fact Recall"),
    ]
    for key, label in labels:
        val = aggregate_scores.get(key)
        if val is not None:
            print(f"  {label:28} {val}")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline metrics")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip answer generation and LLM judge (faster, no API cost)",
    )
    parser.add_argument(
        "--source-type",
        default=None,
        help="Loose filter: keep rows where source_types contains this value",
    )
    parser.add_argument(
        "--include-multi-source",
        action="store_true",
        help="Include questions tagged confluence+jira/slack etc. (unfair if index is confluence-only)",
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help=f"Rebuild {DEFAULT_DATASET} from {QUESTIONS_SOURCE} (confluence only)",
    )
    args = parser.parse_args()

    if args.build_dataset:
        if not os.path.exists(QUESTIONS_SOURCE):
            print(f"Source not found: {QUESTIONS_SOURCE}")
            sys.exit(1)
        build_eval_dataset(
            source_type=args.source_type or "confluence",
            confluence_only=not args.include_multi_source,
        )
        return

    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}")
        print(f"Run: python evaluate.py --build-dataset  (from {QUESTIONS_SOURCE})")
        sys.exit(1)

    confluence_only = (
        not args.include_multi_source
        and args.dataset == DEFAULT_DATASET
        and args.source_type is None
    )
    rows = load_dataset(
        args.dataset,
        source_type=args.source_type,
        confluence_only=confluence_only,
    )
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("Dataset is empty.")
        sys.exit(1)

    if not args.retrieval_only and not OPENROUTER_API_KEY:
        print(
            "Warning: OPENROUTER_API_KEY not set — running retrieval-only.\n"
            "Add key to .env for contextual recall/precision/relevancy metrics."
        )
        args.retrieval_only = True

    print(f"Evaluating {len(rows)} question(s) from {args.dataset}")
    if confluence_only:
        print("Filter: confluence-only (source_types == ['confluence'])")
    if args.retrieval_only:
        print("Mode: retrieval-only (no generation / LLM judge)")
    else:
        print(f"Judge model: {JUDGE_MODEL}")

    index, chunks, model, reranker, bm25 = ensure_loaded()

    results = []
    for row in rows:
        print(f"\nEvaluating {row.get('question_id', '?')}...")
        results.append(
            evaluate_one(
                row, index, chunks, model, reranker, bm25,
                retrieval_only=args.retrieval_only,
            )
        )

    agg = aggregate(results)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "source_type_filter": args.source_type,
        "n_questions": len(results),
        "retrieval_only": args.retrieval_only,
        "judge_model": JUDGE_MODEL if not args.retrieval_only else None,
        "aggregate": agg,
        "per_question": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_report(results, agg)
    print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
