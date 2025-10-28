import json, time, re
from pathlib import Path
from chain import load_chain

ask = load_chain()
items = [json.loads(s) for s in Path("eval/qa.jsonl").read_text().splitlines()]

ok = 0
for it in items:
    t0 = time.time()
    out = ask(it["q"])
    ans = out["answer"].lower()
    if it["ref"] == "REFUSE":
        passed = "not enough info" in ans
    else:
        regex = it["ref"].lower()
        passed = bool(re.search(regex, ans))
    ok += int(passed)
    print(f"Q: {it['q']}\nA: {out['answer']}\nPASS: {passed} | Lat: {out['latency']}s | Src: {out['sources']}\n---")
print(f"Score: {ok}/{len(items)}")
