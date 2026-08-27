#!/usr/bin/env python3
"""
RAG evaluation runner for HomeFinder SF.

Runs every question in eval/questions.py through the LangGraph pipeline,
scores each answer with GPT-4o, and writes results to eval/results/.

Usage:
  python eval/run_eval.py
  python eval/run_eval.py --ids Q01 Q05 Q12     # run specific questions
  python eval/run_eval.py --tags neighborhood    # run by tag
  python eval/run_eval.py --no-judge            # skip GPT-4o scoring (faster)
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.questions import QUESTIONS
from eval.metrics import score_answer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the SF HomeFinder RAG pipeline")
    parser.add_argument("--ids",      nargs="+", help="Run only these question IDs (e.g. Q01 Q03)")
    parser.add_argument("--tags",     nargs="+", help="Run only questions with these tags")
    parser.add_argument("--no-judge", action="store_true", help="Skip GPT-4o judge scoring")
    parser.add_argument("--output",   default="eval/results", help="Directory to write results")
    return parser.parse_args()


def filter_questions(ids=None, tags=None):
    qs = QUESTIONS
    if ids:
        qs = [q for q in qs if q["id"] in ids]
    if tags:
        qs = [q for q in qs if any(t in q["tags"] for t in tags)]
    return qs


def run_question(graph, question_text: str, thread_id: str) -> dict:
    """Invoke the LangGraph pipeline and return raw result fields."""
    start = time.perf_counter()
    result = graph.invoke(
        {"question": question_text, "retry_count": 0},
        config={"configurable": {"thread_id": thread_id}},
    )
    latency = round(time.perf_counter() - start, 2)
    return {
        "answer": result.get("answer", ""),
        "cited_ids": result.get("cited_listing_ids", []),
        "documents": result.get("relevant_docs") or result.get("documents", []),
        "latency_s": latency,
    }


def main():
    args = parse_args()
    questions = filter_questions(ids=args.ids, tags=args.tags)

    if not questions:
        print("No questions matched the filters.")
        sys.exit(0)

    print(f"Loading RAG pipeline...")
    from src.graph.build import build_graph
    graph = build_graph()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"eval_{run_id}.json"
    summary_path = output_dir / f"eval_{run_id}_summary.txt"

    results = []
    total = len(questions)

    print(f"\nRunning {total} question(s)...\n{'─' * 60}")

    for i, q in enumerate(questions, 1):
        qid      = q["id"]
        question = q["question"]
        tags     = q["tags"]
        expect   = q["expect_results"]

        print(f"[{i}/{total}] {qid}: {question[:72]}...")

        try:
            raw = run_question(graph, question, thread_id=f"eval-{run_id}-{qid}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": qid, "error": str(e), "tags": tags})
            continue

        answer      = raw["answer"]
        cited_ids   = raw["cited_ids"]
        n_docs      = len(raw["documents"])
        latency     = raw["latency_s"]
        has_results = len(cited_ids) > 0
        correct_expectation = has_results == expect

        print(f"  Latency: {latency}s | Docs retrieved: {n_docs} | Citations: {len(cited_ids)}")
        print(f"  Expected results: {expect} | Got results: {has_results} {'✓' if correct_expectation else '✗'}")

        row = {
            "id": qid,
            "question": question,
            "tags": tags,
            "answer": answer,
            "cited_ids": cited_ids,
            "n_docs_retrieved": n_docs,
            "n_citations": len(cited_ids),
            "latency_s": latency,
            "expect_results": expect,
            "got_results": has_results,
            "expectation_met": correct_expectation,
        }

        if not args.no_judge and answer:
            print("  Scoring with GPT-4o judge...")
            try:
                score = score_answer(question, answer)
                row["scores"] = {
                    "relevance":    score.relevance,
                    "groundedness": score.groundedness,
                    "helpfulness":  score.helpfulness,
                    "reasoning":    score.reasoning,
                }
                avg = round((score.relevance + score.groundedness + score.helpfulness) / 3, 2)
                row["scores"]["avg"] = avg
                print(f"  Scores — relevance: {score.relevance} | groundedness: {score.groundedness} | helpfulness: {score.helpfulness} | avg: {avg}")
                print(f"  Judge: {score.reasoning}")
            except Exception as e:
                print(f"  Judge error: {e}")
                row["scores"] = None
        else:
            row["scores"] = None

        results.append(row)
        print()

    # ── Write full results JSON ───────────────────────────────────────────
    with open(results_path, "w") as f:
        json.dump({"run_id": run_id, "results": results}, f, indent=2)
    print(f"Results saved → {results_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    scored = [r for r in results if r.get("scores")]
    errored = [r for r in results if "error" in r]
    expectation_met = [r for r in results if r.get("expectation_met")]

    lines = [
        f"{'═' * 60}",
        f"  EVAL SUMMARY  —  run {run_id}",
        f"{'═' * 60}",
        f"  Questions run      : {len(results)}",
        f"  Errors             : {len(errored)}",
        f"  Expectation met    : {len(expectation_met)}/{len(results)}",
    ]

    if scored:
        avg_rel  = round(sum(r["scores"]["relevance"]    for r in scored) / len(scored), 2)
        avg_gnd  = round(sum(r["scores"]["groundedness"] for r in scored) / len(scored), 2)
        avg_help = round(sum(r["scores"]["helpfulness"]  for r in scored) / len(scored), 2)
        avg_all  = round(sum(r["scores"]["avg"]          for r in scored) / len(scored), 2)
        avg_lat  = round(sum(r["latency_s"]              for r in results if "latency_s" in r) / len(results), 2)
        lines += [
            f"",
            f"  GPT-4o Judge scores (avg across {len(scored)} answers):",
            f"    Relevance      : {avg_rel} / 5",
            f"    Groundedness   : {avg_gnd} / 5",
            f"    Helpfulness    : {avg_help} / 5",
            f"    Overall avg    : {avg_all} / 5",
            f"",
            f"  Avg latency      : {avg_lat}s",
        ]

    if scored:
        lines += ["", f"  Per-question scores:"]
        for r in scored:
            s = r["scores"]
            flag = "✓" if r.get("expectation_met") else "✗"
            lines.append(
                f"    {flag} {r['id']}  rel={s['relevance']} gnd={s['groundedness']} "
                f"hlp={s['helpfulness']} avg={s['avg']}  ({r['latency_s']}s)"
            )

    lines.append(f"{'═' * 60}")
    summary = "\n".join(lines)
    print("\n" + summary)

    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
