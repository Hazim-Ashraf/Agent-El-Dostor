"""Evaluation harness (M4).

Runs a golden set of question/expected-citation cases through the agent and
reports groundedness (verification pass rate), citation recall, and refusal
accuracy on out-of-scope questions.

Run inside the container (needs the DB + OpenRouter; costs tokens):
    docker compose run --rm app python eval/run_eval.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from app.agent.loop import run_agent
from app.core.usage import Usage
from app.ingestion.contracts import ingest_contract_bytes
from app.ingestion.legislation import ingest_dir

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")


def load_cases() -> tuple[dict[str, str], list[dict]]:
    contracts: dict[str, str] = {}
    cases: list[dict] = []
    for path in sorted(glob.glob(os.path.join(DATASET_DIR, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        contracts.update(data.get("contracts", {}))
        cases.extend(data.get("cases", []))
    return contracts, cases


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def _ref_match(expected: str, got: set[str]) -> bool:
    e = expected.lower().strip()
    e_digits = _digits(expected)
    for g in got:
        if e == g.lower().strip():
            return True
        # digit-equality handles "Clause 2" (expected) vs "2" (cited); requires the
        # SAME number, so "Art-11" does not match "Art-110".
        if e_digits and e_digits == _digits(g):
            return True
    return False


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agent El-Dostor eval suite.")
    parser.add_argument("--no-setup", action="store_true", help="Skip re-ingesting legislation seeds.")
    args = parser.parse_args()

    contracts_map, cases = load_cases()

    if not args.no_setup:
        print("Setup: ingesting legislation seeds…")
        ingest_dir("data/legislation")

    contract_ids: dict[str, str] = {}
    for key, path in contracts_map.items():
        with open(path, "rb") as fh:
            meta = ingest_contract_bytes(os.path.basename(path), fh.read(), key)
        contract_ids[key] = meta["contract_id"]
        print(f"Setup: contract '{key}' -> {meta['contract_id']} ({meta['n_clauses']} clauses)")

    total = Usage()
    rows: list[dict] = []

    print("\nRunning cases…")
    for case in cases:
        cid = contract_ids.get(case["contract"]) if case.get("contract") else None
        result = run_agent(case["question"], contract_id=cid)
        total.add(Usage(**result["usage"]))

        status = result["verification"]["status"]
        findings = result["verification"]["findings"]
        cits = result.get("citations", [])
        law_got = {c["ref"] for c in cits if c.get("source") == "law" and c.get("ref")}
        clause_got = {c["ref"] for c in cits if c.get("source") == "contract" and c.get("ref")}

        expect = case.get("expect", {})
        refused = findings == 0
        row: dict = {"id": case["id"], "status": status, "findings": findings}

        if expect.get("refuse"):
            row["kind"] = "refuse"
            row["correct"] = refused
        else:
            row["kind"] = "answer"
            row["verified"] = status == "passed"
            law_exp = expect.get("law_refs", [])
            clause_exp = expect.get("clause_refs", [])
            total_exp = len(law_exp) + len(clause_exp)
            if total_exp:
                matched = sum(_ref_match(r, law_got) for r in law_exp) + sum(
                    _ref_match(r, clause_got) for r in clause_exp
                )
                row["citation_recall"] = matched / total_exp
            else:
                row["citation_recall"] = None
            row["correct"] = row["verified"] and (row["citation_recall"] in (None, 1.0))
        rows.append(row)
        print(f"  [{'OK' if row['correct'] else '..'}] {row['id']}: {row}")

    answer_rows = [r for r in rows if r["kind"] == "answer"]
    refuse_rows = [r for r in rows if r["kind"] == "refuse"]
    metrics = {
        "verification_pass_rate": _mean([1.0 if r["verified"] else 0.0 for r in answer_rows]),
        "citation_recall": _mean([r["citation_recall"] for r in answer_rows]),
        "refusal_accuracy": _mean([1.0 if r["correct"] else 0.0 for r in refuse_rows]),
        "total_tokens": total.total_tokens,
        "cost_usd": total.cost_usd,
        "n_answer_cases": len(answer_rows),
        "n_refuse_cases": len(refuse_rows),
    }

    print("\n=== Metrics ===")
    print(f"answerable cases: {len(answer_rows)} · refusal cases: {len(refuse_rows)}")
    print(f"verification pass rate (answerable): {metrics['verification_pass_rate']:.0%}")
    print(f"citation recall (answerable):        {metrics['citation_recall']:.0%}")
    print(f"refusal accuracy:                    {metrics['refusal_accuracy']:.0%}")
    print(f"tokens: {total.total_tokens:,} · cost: ${total.cost_usd:.4f} over {total.calls} model call(s)")

    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_report.json", "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "metrics": metrics}, fh, ensure_ascii=False, indent=2)
    print("\nWrote logs/eval_report.json")


if __name__ == "__main__":
    main()
