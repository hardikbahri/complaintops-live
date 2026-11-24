"""ComplaintOps Live: one daily monitoring run.

1. Fetch the newest consumer complaints (with narratives) published by the CFPB.
2. Keep only the ones not seen in earlier runs.
3. Classify them with the ComplaintOps model (LoRA-tuned DistilBERT, int8 ONNX, from the Hugging Face Hub).
4. Compare each prediction with the product + issue the consumer picked, and measure drift
   against a baseline built from the first live runs.
5. Write the day's predictions, a report, the history table, a trend chart and the README dashboard.

Run from the repo root:  python -m monitor.run
"""
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from monitor.labels import merge_label, merged_classes, true_label

ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = "hb-007/complaintops-distilbert-lora-onnx"
MODEL_DIR = ROOT / ".model"
API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

THRESHOLD = 0.80          # same as ComplaintOps: below this confidence, a person reviews the complaint
FETCH_SIZE = int(os.environ.get("FETCH_SIZE", "1000"))
MAX_NEW = int(os.environ.get("MAX_NEW", "800"))
BASELINE_MIN = 1000       # complaints collected before the drift baseline is frozen
OFFLINE = {"auto_rate": 0.726, "auto_acc": 0.939}   # ComplaintOps test-set results (README of the main repo)
PSI_WATCH, PSI_ALERT = 0.10, 0.25
CONF_BINS = np.linspace(0, 1, 21)
MAX_LEN = 256

DAILY = ROOT / "data" / "daily"
REPORTS = ROOT / "reports"
STATE = ROOT / "state"
HISTORY = ROOT / "history.csv"
HIST_COLS = ["date", "new", "auto_rate", "auto_acc", "overall_acc", "out_of_taxonomy", "psi_class", "psi_conf", "status"]


# ---------- data ----------
def fetch_complaints():
    params = {"size": FETCH_SIZE, "sort": "created_date_desc", "has_narrative": "true",
              "no_aggs": "true", "format": "json"}
    headers = {"User-Agent": "complaintops-live (github.com/hardikbahri/complaintops-live)"}
    last = None
    for attempt in range(4):
        try:
            r = requests.get(API, params=params, headers=headers, timeout=90)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:  # network hiccups or API errors: retry with backoff
            last = e
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"CFPB API unreachable: {last}")
    if isinstance(data, dict):
        data = data.get("hits", {}).get("hits", data.get("results", []))
    rows = [d.get("_source", d) for d in data]
    out = []
    for s in rows:
        text = (s.get("complaint_what_happened") or "").strip()
        if len(text) < 10 or not s.get("complaint_id"):
            continue
        out.append({"complaint_id": str(s["complaint_id"]), "date_received": (s.get("date_received") or "")[:10],
                    "product": s.get("product") or "", "issue": s.get("issue") or "", "text": text})
    return out


def load_seen():
    p = STATE / "seen_ids.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


def save_seen(seen, new_ids):
    ids = sorted(seen | set(new_ids), key=lambda x: int(x) if x.isdigit() else 0)[-30000:]
    (STATE / "seen_ids.json").write_text(json.dumps(ids))


# ---------- model ----------
class Model:
    def __init__(self):
        MODEL_DIR.mkdir(exist_ok=True)
        f = {n: hf_hub_download(MODEL_REPO, n, local_dir=MODEL_DIR)
             for n in ["model_int8.onnx", "tokenizer.json", "classes.json", "bias.json"]}
        self.full = json.load(open(f["classes.json"]))
        self.classes = merged_classes(self.full)
        self.index = {c: i for i, c in enumerate(self.classes)}
        bias = json.load(open(f["bias.json"]))
        self.bias = np.array(bias["bias"] if isinstance(bias, dict) else bias, dtype="float32")
        self.tok = Tokenizer.from_file(f["tokenizer.json"])
        self.tok.enable_truncation(max_length=MAX_LEN)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("[PAD]"), pad_token="[PAD]")
        self.sess = ort.InferenceSession(f["model_int8.onnx"], providers=["CPUExecutionProvider"])

    def proba(self, texts, bs=16):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok.encode_batch(texts[i:i + bs])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            logits = self.sess.run(None, {"input_ids": ids, "attention_mask": mask})[0] + self.bias
            z = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = z / z.sum(axis=1, keepdims=True)
            m = np.zeros((p.shape[0], len(self.classes)), dtype=p.dtype)
            for j, c in enumerate(self.full):
                m[:, self.index[merge_label(c)]] += p[:, j]
            out.append(m)
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)))


# ---------- metrics ----------
def psi(ref, cur):
    ref, cur = np.clip(np.asarray(ref, float), 1e-4, None), np.clip(np.asarray(cur, float), 1e-4, None)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(((cur - ref) * np.log(cur / ref)).sum())


def read_daily(days=None):
    files = sorted(DAILY.glob("*.csv"))
    if days:
        files = files[-days:]
    rows = []
    for f in files:
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def dist(rows, classes):
    counts = np.zeros(len(classes))
    idx = {c: i for i, c in enumerate(classes)}
    for r in rows:
        counts[idx[r["pred"]]] += 1
    return counts


def conf_hist(rows):
    return np.histogram([float(r["confidence"]) for r in rows], CONF_BINS)[0].astype(float)


def summarize(rows):
    n = len(rows)
    if not n:
        return {"new": 0}
    auto = [r for r in rows if r["auto"] == "1"]
    auto_in = [r for r in auto if r["in_taxonomy"] == "1"]
    in_tax = [r for r in rows if r["in_taxonomy"] == "1"]
    return {"new": n,
            "auto_rate": len(auto) / n,
            "auto_acc": (sum(r["correct"] == "1" for r in auto_in) / len(auto_in)) if auto_in else None,
            "overall_acc": (sum(r["correct"] == "1" for r in in_tax) / len(in_tax)) if in_tax else None,
            "out_of_taxonomy": 1 - len(in_tax) / n}


def baseline(classes, today):
    p = STATE / "baseline.json"
    if p.exists():
        return json.loads(p.read_text())
    rows = read_daily()
    if len(rows) < BASELINE_MIN:
        return {"pending": len(rows)}
    b = {"frozen_on": today, "n": len(rows),
         "class_counts": dist(rows, classes).tolist(), "conf_hist": conf_hist(rows).tolist(),
         "summary": summarize(rows)}
    p.write_text(json.dumps(b, indent=1))
    return b


def status_of(s, psi_class, psi_conf):
    if s["new"] < 30:
        return "LOW VOLUME"
    acc = s.get("auto_acc")
    if (psi_class or 0) > PSI_ALERT or (psi_conf or 0) > PSI_ALERT or (acc is not None and acc < OFFLINE["auto_acc"] - 0.08):
        return "ALERT"
    if (psi_class or 0) > PSI_WATCH or (psi_conf or 0) > PSI_WATCH or (acc is not None and acc < OFFLINE["auto_acc"] - 0.04):
        return "WATCH"
    return "OK"


def pct(x):
    return "n/a" if x is None or x == "" else f"{float(x) * 100:.1f}%"


def num(x):
    return "n/a" if x is None or x == "" else f"{float(x):.3f}"


# ---------- outputs ----------
def write_history(today, s, psi_class, psi_conf, status):
    rows = []
    if HISTORY.exists():
        with open(HISTORY) as fh:
            rows = [r for r in csv.DictReader(fh) if r["date"] != today]
    r4 = lambda x: "" if x is None or x == "" else round(float(x), 4)
    rows.append({"date": today, "new": s["new"], "auto_rate": r4(s.get("auto_rate")), "auto_acc": r4(s.get("auto_acc")),
                 "overall_acc": r4(s.get("overall_acc")), "out_of_taxonomy": r4(s.get("out_of_taxonomy")),
                 "psi_class": r4(psi_class), "psi_conf": r4(psi_conf), "status": status})
    with open(HISTORY, "w", newline="") as fh:
        w = csv.DictWriter(fh, HIST_COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def chart(hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = [r for r in hist if int(r["new"]) > 0]
    if len(h) < 2:
        return False
    d = [dt.date.fromisoformat(r["date"]) for r in h]
    f = lambda k: [float(r[k]) * 100 if r[k] not in ("", None) else np.nan for r in h]
    g = lambda k: [float(r[k]) if r[k] not in ("", None) else np.nan for r in h]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(d, f("auto_acc"), marker="o", ms=3, color="#1f6feb", label="Accuracy on auto-answered")
    a1.plot(d, f("auto_rate"), marker="o", ms=3, color="#2da44e", label="Auto-answer rate")
    a1.axhline(OFFLINE["auto_acc"] * 100, ls="--", lw=1, color="#1f6feb", alpha=.5)
    a1.axhline(OFFLINE["auto_rate"] * 100, ls="--", lw=1, color="#2da44e", alpha=.5)
    a1.set_ylabel("%")
    a1.set_ylim(0, 100)
    a1.set_title("ComplaintOps on live CFPB complaints (dashed = offline test set)")
    a1.legend(loc="lower left", fontsize=8)
    a2.plot(d, g("psi_class"), marker="o", ms=3, color="#8250df", label="PSI, predicted class mix")
    a2.plot(d, g("psi_conf"), marker="o", ms=3, color="#bf8700", label="PSI, confidence")
    a2.axhline(PSI_WATCH, ls=":", lw=1, color="grey")
    a2.axhline(PSI_ALERT, ls="--", lw=1, color="#cf222e")
    a2.set_ylabel("PSI vs baseline")
    a2.legend(loc="upper left", fontsize=8)
    import matplotlib.dates as mdates
    a2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(d) // 8)))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(REPORTS / "trend.png", dpi=120)
    plt.close(fig)
    return True


def movers(rows, base, classes, k=3):
    if not rows or "class_counts" not in base:
        return []
    cur = dist(rows, classes)
    cur, ref = cur / cur.sum(), np.array(base["class_counts"]) / sum(base["class_counts"])
    delta = cur - ref
    return [(classes[i], ref[i], cur[i]) for i in np.argsort(-delta)[:k] if delta[i] > 0.01]


def write_report(today, s, base, psi_class, psi_conf, status, top, rows):
    lines = [f"# ComplaintOps Live: {today}", "",
             f"**Status: {status}**", "",
             "| Metric | Today | Offline test set |", "| --- | --- | --- |",
             f"| New complaints classified | {s['new']} | |",
             f"| Auto-answered (confidence >= {THRESHOLD}) | {pct(s.get('auto_rate'))} | {pct(OFFLINE['auto_rate'])} |",
             f"| Accuracy on auto-answered | {pct(s.get('auto_acc'))} | {pct(OFFLINE['auto_acc'])} |",
             f"| Accuracy on all (top-1) | {pct(s.get('overall_acc'))} | |",
             f"| Outside the model's 24 classes | {pct(s.get('out_of_taxonomy'))} | |",
             f"| PSI, predicted class mix | {num(psi_class)} | < {PSI_WATCH} stable |",
             f"| PSI, confidence | {num(psi_conf)} | < {PSI_WATCH} stable |", ""]
    if "pending" in base:
        lines += [f"Drift baseline still building: {base['pending']}/{BASELINE_MIN} complaints collected.", ""]
    if top:
        lines += ["Classes growing most vs baseline:", ""]
        lines += [f"- {c}: {r * 100:.1f}% -> {cu * 100:.1f}%" for c, r, cu in top]
        lines.append("")
    if rows:
        lines += ["Accuracy is measured against the product and issue the consumer selected when filing. "
                  "Complaints are matched to the training taxonomy with the same mapping used in training.", ""]
    (REPORTS / f"{today}.md").write_text("\n".join(lines))


def write_readme(today, hist, base, has_chart):
    last7 = read_daily(7)
    s7 = summarize(last7)
    recent = [r for r in hist][-7:][::-1]
    latest = hist[-1]
    block = ["<!-- LIVE:START -->",
             f"**Last run:** {today} | **Status:** {latest['status']} | "
             f"**Complaints monitored so far:** {sum(int(r['new']) for r in hist):,}", ""]
    if s7["new"]:
        block += ["**Last 7 runs**", "",
                  "| | Live | Offline test set |", "| --- | --- | --- |",
                  f"| Complaints | {s7['new']} | |",
                  f"| Auto-answered | {pct(s7.get('auto_rate'))} | {pct(OFFLINE['auto_rate'])} |",
                  f"| Accuracy on auto-answered | {pct(s7.get('auto_acc'))} | {pct(OFFLINE['auto_acc'])} |", ""]
    if "pending" in base:
        block += [f"Drift baseline: building ({base['pending']}/{BASELINE_MIN} complaints).", ""]
    else:
        block += [f"Drift baseline: {base['n']:,} complaints, frozen on {base['frozen_on']}.", ""]
    if has_chart:
        block += ["![Trend](reports/trend.png)", ""]
    block += ["| Date | New | Auto-answered | Accuracy (auto) | PSI class mix | Status |",
              "| --- | --- | --- | --- | --- | --- |"]
    block += [f"| [{r['date']}](reports/{r['date']}.md) | {r['new']} | {pct(r['auto_rate'])} | {pct(r['auto_acc'])} | "
              f"{num(r['psi_class'])} | {r['status']} |" for r in recent]
    block.append("<!-- LIVE:END -->")
    readme = (ROOT / "README.md").read_text()
    a, b = readme.index("<!-- LIVE:START -->"), readme.index("<!-- LIVE:END -->") + len("<!-- LIVE:END -->")
    (ROOT / "README.md").write_text(readme[:a] + "\n".join(block) + readme[b:])


def main():
    today = os.environ.get("RUN_DATE") or dt.datetime.utcnow().date().isoformat()
    for p in (DAILY, REPORTS, STATE):
        p.mkdir(parents=True, exist_ok=True)

    seen = load_seen()
    fetched = fetch_complaints()
    new = [c for c in fetched if c["complaint_id"] not in seen][:MAX_NEW]
    print(f"fetched {len(fetched)}, new {len(new)}")

    model = Model()
    base = baseline(model.classes, today)   # built only from earlier runs
    rows = []
    if new:
        P = model.proba([c["text"] for c in new])
        for c, p in zip(new, P):
            j = int(p.argmax())
            truth = true_label(c["product"], c["issue"])
            in_tax = truth in model.index
            rows.append({"complaint_id": c["complaint_id"], "date_received": c["date_received"],
                         "true_label": truth, "in_taxonomy": "1" if in_tax else "0", "pred": model.classes[j],
                         "confidence": f"{p[j]:.4f}", "auto": "1" if p[j] >= THRESHOLD else "0",
                         "correct": "1" if in_tax and model.classes[j] == truth else "0"})
        with open(DAILY / f"{today}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        save_seen(seen, [c["complaint_id"] for c in new])

    s = summarize(rows)
    psi_class = psi_conf = None
    if rows and "class_counts" in base:
        psi_class = psi(base["class_counts"], dist(rows, model.classes))
        psi_conf = psi(base["conf_hist"], conf_hist(rows))
    status = status_of(s, psi_class, psi_conf) if rows else "NO NEW DATA"
    top = movers(rows, base, model.classes)

    hist = write_history(today, s, psi_class, psi_conf, status)
    write_report(today, s, base, psi_class, psi_conf, status, top, rows)
    has_chart = chart(hist)
    write_readme(today, hist, base, has_chart)

    msg = (f"Daily run {today}: {s['new']} complaints, auto-answered {pct(s.get('auto_rate'))}, "
           f"accuracy {pct(s.get('auto_acc'))}, status {status}") if rows else f"Daily run {today}: no new complaints published"
    (ROOT / ".commit_msg").write_text(msg)
    print(msg)


if __name__ == "__main__":
    sys.exit(main())
