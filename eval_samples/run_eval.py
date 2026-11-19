import json
import sys
from pathlib import Path
from statistics import mean

# Answers can legitimately contain non-ASCII characters (the sample docs use
# one), and Windows terminals default to a codepage that can't print them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chain import load_chain

EVAL_FILE = Path(__file__).resolve().parent / "qa.jsonl"
REPORT_FILE = Path(__file__).resolve().parent / "report.md"


def check(item, answer: str, sources=None) -> bool:
    ans = answer.lower()
    if item["type"] == "refuse":
        # A real refusal must not cite sources: the score-threshold gate in
        # chain.py returns an empty sources list when it refuses.
        return "not enough info" in ans and not sources
    if item["type"] == "contains":
        return item["text"].lower() in ans
    if item["type"] == "keywords":
        return any(kw.lower() in ans for kw in item["any_of"])
    raise ValueError(f"Unknown eval type: {item['type']}")


def run():
    ask = load_chain()
    items = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]

    rows = []
    for item in items:
        out = ask(item["q"])
        passed = check(item, out["answer"], out["sources"])
        rows.append(
            {
                "question": item["q"],
                "answer": out["answer"],
                "sources": out["sources"],
                "latency": out["latency"],
                "passed": passed,
            }
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] {item['q']}\n  -> {out['answer']}\n"
            f"  ({out['latency']}s, sources: {out['sources']})\n"
        )

    passed_count = sum(r["passed"] for r in rows)
    latencies = [r["latency"] for r in rows]
    print(
        f"Score: {passed_count}/{len(rows)} | avg latency: {mean(latencies):.3f}s "
        f"| max latency: {max(latencies):.3f}s"
    )

    write_report(rows, passed_count)
    return rows


def write_report(rows, passed_count):
    lines = [
        "# Eval report",
        "",
        f"**Score:** {passed_count}/{len(rows)}  ",
        f"**Avg latency:** {mean(r['latency'] for r in rows):.3f}s  ",
        f"**Max latency:** {max(r['latency'] for r in rows):.3f}s",
        "",
        "| # | Question | Result | Latency | Sources |",
        "|---|----------|--------|---------|---------|",
    ]
    for i, r in enumerate(rows, 1):
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        sources = ", ".join(r["sources"]) or "—"
        lines.append(f"| {i} | {r['question']} | {status} | {r['latency']}s | {sources} |")
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {REPORT_FILE}")


if __name__ == "__main__":
    run()
