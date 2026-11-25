"""Daily model evaluation: run the published ComplaintOps model on a fresh sample of held-out complaints.

The CFPB no longer publishes complaint text, so live inference is not possible. Instead, every day the real int8 ONNX
model from the Hugging Face Hub classifies 500 complaints drawn at random (seeded by the date) from the held-out test
split: the newest 15% of the data, never seen in training. It records accuracy, how many complaints the confidence gate
would answer automatically, the accuracy of those answers, and CPU latency, so a broken model upload, a tokenizer
mismatch or a slow runtime shows up within a day.

Run from the repo root (after monitor.run):  python -m monitor.evaluate
"""
import csv
import datetime as dt
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from monitor.labels import merge_label, merged_classes

ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = "hb-007/complaintops-distilbert-lora-onnx"
TEST_SET = ROOT / "data" / "eval" / "test.jsonl.gz"
HISTORY = ROOT / "eval_history.csv"
REPORTS = ROOT / "reports" / "eval"
SAMPLE = int(os.environ.get("EVAL_SAMPLE", "500"))
THRESHOLD = 0.80
MAX_LEN = 256
OFFLINE = {"auto_rate": 0.726, "auto_acc": 0.939}  # full test split, from the ComplaintOps README
COLS = ["date", "n", "accuracy", "auto_rate", "auto_accuracy", "ms_per_complaint_batched", "ms_single_p50", "ms_single_p95"]


class Model:
    """Same steps as complaintops.serve_onnx: truncate to 256 tokens, add the per-class bias, softmax, merge classes."""

    def __init__(self):
        files = {n: hf_hub_download(MODEL_REPO, n) for n in
                 ["model_int8.onnx", "tokenizer.json", "classes.json", "bias.json"]}
        self.full = json.load(open(files["classes.json"]))
        self.classes = merged_classes(self.full)
        self.index = {c: i for i, c in enumerate(self.classes)}
        self.bias = np.array(json.load(open(files["bias.json"]))["bias"], dtype="float32")
        self.tok = Tokenizer.from_file(files["tokenizer.json"])
        self.tok.enable_truncation(max_length=MAX_LEN)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("[PAD]"), pad_token="[PAD]")
        self.sess = ort.InferenceSession(files["model_int8.onnx"], providers=["CPUExecutionProvider"])

    def proba(self, texts, batch_size=16):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[i:i + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            logits = self.sess.run(None, {"input_ids": ids, "attention_mask": mask})[0] + self.bias
            z = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = z / z.sum(axis=1, keepdims=True)
            m = np.zeros((p.shape[0], len(self.classes)), dtype=p.dtype)
            for j, c in enumerate(self.full):
                m[:, self.index[merge_label(c)]] += p[:, j]
            out.append(m)
        return np.concatenate(out)


def load_test():
    with gzip.open(TEST_SET, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def evaluate(day: str):
    rows = load_test()
    rng = np.random.default_rng(int(day.replace("-", "")))  # a different, reproducible sample every day
    sample = [rows[i] for i in rng.choice(len(rows), size=min(SAMPLE, len(rows)), replace=False)]
    texts = [r["text"] for r in sample]
    truth = [merge_label(r["label"]) for r in sample]

    model = Model()
    model.proba(texts[:16])  # warm-up, so the timing measures steady state
    t0 = time.perf_counter()
    P = model.proba(texts)
    batched_ms = (time.perf_counter() - t0) * 1000 / len(texts)
    single = []
    for t in texts[:50]:  # one request at a time, as the live API serves them
        t1 = time.perf_counter()
        model.proba([t], batch_size=1)
        single.append((time.perf_counter() - t1) * 1000)

    pred = [model.classes[j] for j in P.argmax(1)]
    conf = P.max(1)
    correct = np.array([p == t for p, t in zip(pred, truth)])
    auto = conf >= THRESHOLD
    return {"date": day, "n": len(sample), "accuracy": round(float(correct.mean()), 4),
            "auto_rate": round(float(auto.mean()), 4),
            "auto_accuracy": round(float(correct[auto].mean()), 4) if auto.any() else "",
            "ms_per_complaint_batched": round(batched_ms, 1),
            "ms_single_p50": round(float(np.percentile(single, 50)), 1),
            "ms_single_p95": round(float(np.percentile(single, 95)), 1)}


def save(rec):
    rows = []
    if HISTORY.exists():
        with open(HISTORY) as fh:
            rows = [r for r in csv.DictReader(fh) if r["date"] != rec["date"]]
    rows = sorted(rows + [rec], key=lambda r: r["date"])
    with open(HISTORY, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def pct(x):
    return "n/a" if x in ("", None) else f"{float(x) * 100:.1f}%"


def report(rec):
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = [f"# Model evaluation: {rec['date']}", "",
             f"The published int8 ONNX model classified {rec['n']} held-out complaints sampled for this date.", "",
             "| Metric | Today | Full test split |", "| --- | --- | --- |",
             f"| Accuracy (top-1, 24 classes) | {pct(rec['accuracy'])} | |",
             f"| Answered automatically (confidence >= {THRESHOLD}) | {pct(rec['auto_rate'])} | {pct(OFFLINE['auto_rate'])} |",
             f"| Accuracy of automatic answers | {pct(rec['auto_accuracy'])} | {pct(OFFLINE['auto_acc'])} |",
             f"| CPU latency per complaint, batched | {rec['ms_per_complaint_batched']} ms | |",
             f"| CPU latency, single request p50 / p95 | {rec['ms_single_p50']} / {rec['ms_single_p95']} ms | |", "",
             "Runs on a GitHub Actions runner (4 vCPU, no GPU).", ""]
    (REPORTS / f"{rec['date']}.md").write_text("\n".join(lines))


def chart(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    if len(rows) < 2:
        return False
    d = [dt.date.fromisoformat(r["date"]) for r in rows]
    f = lambda k: [float(r[k]) * 100 if r[k] not in ("", None) else np.nan for r in rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(d, f("auto_accuracy"), marker="o", ms=3, color="#1f6feb", label="Accuracy of automatic answers")
    a1.plot(d, f("auto_rate"), marker="o", ms=3, color="#2da44e", label="Answered automatically")
    a1.plot(d, f("accuracy"), marker="o", ms=3, color="#8c959f", label="Accuracy, all")
    a1.axhline(OFFLINE["auto_acc"] * 100, ls="--", lw=1, color="#1f6feb", alpha=.5)
    a1.axhline(OFFLINE["auto_rate"] * 100, ls="--", lw=1, color="#2da44e", alpha=.5)
    a1.set_ylim(0, 100)
    a1.set_ylabel("%")
    a1.set_title("Daily evaluation of the published model (dashed = full test split)")
    a1.legend(loc="lower left", fontsize=8)
    a2.plot(d, [float(r["ms_single_p50"]) for r in rows], marker="o", ms=3, color="#bf8700", label="single request p50")
    a2.plot(d, [float(r["ms_per_complaint_batched"]) for r in rows], marker="o", ms=3, color="#8250df", label="batched, per complaint")
    a2.set_ylabel("ms on CPU")
    a2.set_ylim(bottom=0)
    a2.legend(loc="upper left", fontsize=8)
    a2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(d) // 8)))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "eval_trend.png", dpi=120)
    plt.close(fig)
    return True


def readme(rows, has_chart):
    last = rows[-1]
    block = ["<!-- EVAL:START -->",
             f"**Latest ({last['date']}):** {last['n']} held-out complaints, accuracy {pct(last['accuracy'])}, "
             f"answered automatically {pct(last['auto_rate'])} at {pct(last['auto_accuracy'])} accuracy, "
             f"{last['ms_single_p50']} ms per request (p50, CPU).", ""]
    if has_chart:
        block += ["![Evaluation trend](reports/eval_trend.png)", ""]
    block += ["| Date | Accuracy | Auto-answered | Accuracy (auto) | p50 ms |", "| --- | --- | --- | --- | --- |"]
    block += [f"| [{r['date']}](reports/eval/{r['date']}.md) | {pct(r['accuracy'])} | {pct(r['auto_rate'])} | "
              f"{pct(r['auto_accuracy'])} | {r['ms_single_p50']} |" for r in rows[-7:][::-1]]
    block.append("<!-- EVAL:END -->")
    text = (ROOT / "README.md").read_text()
    a, b = text.index("<!-- EVAL:START -->"), text.index("<!-- EVAL:END -->") + len("<!-- EVAL:END -->")
    (ROOT / "README.md").write_text(text[:a] + "\n".join(block) + text[b:])


def main():
    day = os.environ.get("RUN_DATE") or dt.datetime.utcnow().date().isoformat()
    if not TEST_SET.exists():  # the held-out split is added separately; until then the daily run is unaffected
        print(f"skipping evaluation: {TEST_SET.relative_to(ROOT)} not found")
        return
    rec = evaluate(day)
    rows = save(rec)
    report(rec)
    readme(rows, chart(rows))
    msg_file = ROOT / ".commit_msg"
    extra = f" | model eval: accuracy {pct(rec['accuracy'])}, auto {pct(rec['auto_rate'])} at {pct(rec['auto_accuracy'])}"
    msg_file.write_text((msg_file.read_text().strip() if msg_file.exists() else f"Daily run {day}") + extra)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
