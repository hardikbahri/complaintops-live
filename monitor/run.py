"""ComplaintOps Live: one daily monitoring run.

The CFPB no longer publishes complaint narratives, so the model cannot be run on new complaints.
What is still published every day is each complaint's product and issue, which are exactly the
labels ComplaintOps predicts. This job watches those labels:

1. Count the complaints received on the target day (15 days ago, so late publications have landed),
   by product and issue, from the CFPB public API.
2. Map each product + issue onto the model's 24 classes (same mapping as training, taxonomy read
   from the model's classes.json on the Hugging Face Hub).
3. Measure coverage (share of complaints the model has a class for) and label drift (PSI of the
   class mix against a 28-day reference window).
4. Write the day's counts, a report, the history table, a trend chart and the README dashboard.

Run from the repo root:  python -m monitor.run
"""
import csv
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import hf_hub_download

from monitor.labels import merged_classes, true_label

ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = "hb-007/complaintops-distilbert-lora-onnx"
API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
HEADERS = {"User-Agent": "complaintops-live monitor (github.com/hardikbahri/complaintops-live)"}

LAG_DAYS = 15        # complaints are published after the company responds (up to ~15 days)
BASELINE_DAYS = 28   # reference window, ending just before the first monitored day
BACKFILL_DAYS = 14   # history filled in on the very first run
PSI_WATCH, PSI_ALERT = 0.10, 0.25
COV_WATCH, COV_ALERT = 0.05, 0.10
MIN_VOLUME = 200
OUTSIDE = "Outside the model's classes"

DAILY = ROOT / "data" / "daily"
REPORTS = ROOT / "reports"
STATE = ROOT / "state"
HISTORY = ROOT / "history.csv"
HIST_COLS = ["date", "complaints", "coverage", "psi_labels", "top_class", "top_class_share", "status"]


# ---------- CFPB API ----------
def api(params):
    last = None
    for attempt in range(5):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"CFPB API failed for {params}: {last}")


def buckets(data, field):
    agg = (data.get("aggregations") or {}).get(field, {})
    return {b["key"]: b["doc_count"] for b in agg.get(field, agg).get("buckets", [])}


def counts_for(start, end):
    """Complaint counts by (product, issue) for date_received in [start, end] (inclusive)."""
    window = {"size": 0, "date_received_min": start, "date_received_max": end}
    top = api(window)
    total = top["hits"]["total"]["value"] if isinstance(top["hits"]["total"], dict) else top["hits"]["total"]
    products = buckets(top, "product")
    with ThreadPoolExecutor(4) as ex:
        per = dict(zip(products, ex.map(lambda p: buckets(api({**window, "product": p}), "issue"), products)))
    pairs = {(p, i): n for p, issues in per.items() for i, n in issues.items()}
    return total, pairs


# ---------- metrics ----------
def class_counts(pairs, classes):
    idx = {c: k for k, c in enumerate(classes)}
    v = np.zeros(len(classes) + 1)
    for (p, i), n in pairs.items():
        v[idx.get(true_label(p, i), len(classes))] += n
    return v


def psi(ref, cur):
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def status_of(n, cov, base_cov, p):
    if n < MIN_VOLUME:
        return "LOW VOLUME"
    drop = base_cov - cov
    if p > PSI_ALERT or drop > COV_ALERT:
        return "ALERT"
    if p > PSI_WATCH or drop > COV_WATCH:
        return "WATCH"
    return "OK"


def show(label):
    return label.replace(" | ", " / ")  # keeps markdown tables intact


def pct(x):
    return "n/a" if x in (None, "") else f"{float(x) * 100:.1f}%"


# ---------- one day ----------
def analyse(day, classes, base):
    total, pairs = counts_for(day, day)
    cc = class_counts(pairs, classes)
    n = cc.sum()
    cov = 1 - cc[-1] / n if n else 0.0
    base_v = np.array(base["class_counts"])
    p = psi(base_v, cc) if n else 0.0
    share = cc / n if n else cc
    base_share = base_v / base_v.sum()
    top = int(np.argmax(share[:-1]))
    return {"date": day, "total": total, "n": int(n), "pairs": pairs, "cc": cc, "coverage": cov, "psi": p,
            "share": share, "base_share": base_share, "top": classes[top], "top_share": float(share[top]),
            "status": status_of(n, cov, base["coverage"], p)}


def write_day(r, classes):
    labels = classes + [OUTSIDE]
    with open(DAILY / f"{r['date']}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["product", "issue", "model_class", "complaints"])
        for (prod, iss), cnt in sorted(r["pairs"].items(), key=lambda x: -x[1]):
            lab = true_label(prod, iss)
            w.writerow([prod, iss, lab if lab in classes else OUTSIDE, cnt])

    order = np.argsort(-(r["share"] - r["base_share"]))
    rising = [(labels[k], r["base_share"][k], r["share"][k]) for k in order[:3] if r["share"][k] - r["base_share"][k] > 0.005]
    outside = {}
    for (prod, iss), cnt in r["pairs"].items():
        lab = true_label(prod, iss)
        if lab not in classes:
            outside[lab] = outside.get(lab, 0) + cnt
    out_top = sorted(outside.items(), key=lambda x: -x[1])[:5]

    lines = [f"# ComplaintOps Live: complaints received {r['date']}", "", f"**Status: {r['status']}**", "",
             "| Metric | This day | Reference window |", "| --- | --- | --- |",
             f"| Complaints received | {r['total']:,} | |",
             f"| Covered by the model's 24 classes | {pct(r['coverage'])} | {pct(1 - r['base_share'][-1])} |",
             f"| PSI of class mix vs reference | {r['psi']:.3f} | < {PSI_WATCH} stable |",
             f"| Largest class | {show(r['top'])} ({pct(r['top_share'])}) | |", ""]
    if rising:
        lines += ["**Classes growing most vs reference**", ""]
        lines += [f"- {show(c)}: {b * 100:.1f}% -> {s * 100:.1f}%" for c, b, s in rising] + [""]
    if out_top:
        lines += ["**Largest product + issue pairs the model has no class for**", ""]
        lines += [f"- {show(lab)}: {cnt:,}" for lab, cnt in out_top] + [""]
    lines += ["Counts come from the CFPB public API for this receipt date and can still change slightly "
              "if late complaints are published.", ""]
    (REPORTS / f"{r['date']}.md").write_text("\n".join(lines))


def load_history():
    if not HISTORY.exists():
        return []
    with open(HISTORY) as fh:
        return list(csv.DictReader(fh))


def save_history(rows):
    rows = sorted({r["date"]: r for r in rows}.values(), key=lambda r: r["date"])
    with open(HISTORY, "w", newline="") as fh:
        w = csv.DictWriter(fh, HIST_COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def hist_row(r):
    return {"date": r["date"], "complaints": r["total"], "coverage": round(r["coverage"], 4),
            "psi_labels": round(r["psi"], 4), "top_class": r["top"], "top_class_share": round(r["top_share"], 4),
            "status": r["status"]}


def chart(hist, base_cov):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    if len(hist) < 2:
        return False
    d = [dt.date.fromisoformat(r["date"]) for r in hist]
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1.4]})
    a0.plot(d, [float(r["coverage"]) * 100 for r in hist], marker="o", ms=3, color="#1f6feb")
    a0.axhline(base_cov * 100, ls="--", lw=1, color="#1f6feb", alpha=.5, label="Reference window")
    a0.set_ylabel("Coverage %")
    a0.set_title("Complaints the ComplaintOps model has a class for, and label drift")
    a0.legend(loc="lower left", fontsize=8)
    a1.plot(d, [float(r["psi_labels"]) for r in hist], marker="o", ms=3, color="#8250df")
    a1.axhline(PSI_WATCH, ls=":", lw=1, color="grey", label=f"Watch {PSI_WATCH}")
    a1.axhline(PSI_ALERT, ls="--", lw=1, color="#cf222e", label=f"Alert {PSI_ALERT}")
    a1.set_ylabel("PSI, class mix")
    a1.set_ylim(bottom=0)
    a1.legend(loc="upper left", fontsize=8)
    a2.bar(d, [int(r["complaints"]) for r in hist], color="#8c959f")
    a2.set_ylabel("Complaints")
    a2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(d) // 8)))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(REPORTS / "trend.png", dpi=120)
    plt.close(fig)
    return True


def write_readme(hist, base, has_chart, run_day):
    latest = hist[-1]
    recent = hist[-7:][::-1]
    block = ["<!-- LIVE:START -->",
             f"**Last run:** {run_day} | **Latest receipt date checked:** {latest['date']} | **Status:** {latest['status']}", "",
             f"Reference window: complaints received {base['start']} to {base['end']} "
             f"({base['n']:,} complaints, {pct(base['coverage'])} covered by the model's classes).", ""]
    if has_chart:
        block += ["![Trend](reports/trend.png)", ""]
    block += ["| Received | Complaints | Coverage | PSI class mix | Largest class | Status |",
              "| --- | --- | --- | --- | --- | --- |"]
    block += [f"| [{r['date']}](reports/{r['date']}.md) | {int(r['complaints']):,} | {pct(r['coverage'])} | "
              f"{float(r['psi_labels']):.3f} | {show(r['top_class'])} ({pct(r['top_class_share'])}) | {r['status']} |" for r in recent]
    block.append("<!-- LIVE:END -->")
    readme = (ROOT / "README.md").read_text()
    a, b = readme.index("<!-- LIVE:START -->"), readme.index("<!-- LIVE:END -->") + len("<!-- LIVE:END -->")
    (ROOT / "README.md").write_text(readme[:a] + "\n".join(block) + readme[b:])


def main():
    run_day = dt.date.fromisoformat(os.environ["RUN_DATE"]) if os.environ.get("RUN_DATE") else dt.datetime.utcnow().date()
    target = run_day - dt.timedelta(days=LAG_DAYS)
    for p in (DAILY, REPORTS, STATE):
        p.mkdir(parents=True, exist_ok=True)

    classes = merged_classes(json.load(open(hf_hub_download(MODEL_REPO, "classes.json"))))
    hist = load_history()
    if not hist:  # first run: fill two weeks of history so the trend means something from day one
        days = [target - dt.timedelta(days=k) for k in range(BACKFILL_DAYS - 1, -1, -1)]
    else:
        last = dt.date.fromisoformat(hist[-1]["date"])  # catch up on missed days, at most a week
        days = [last + dt.timedelta(days=k) for k in range(1, (target - last).days + 1)][-7:]

    bpath = STATE / "baseline.json"
    if bpath.exists():
        base = json.loads(bpath.read_text())
    else:
        end = days[0] - dt.timedelta(days=1)
        start = end - dt.timedelta(days=BASELINE_DAYS - 1)
        _, pairs = counts_for(start.isoformat(), end.isoformat())
        cc = class_counts(pairs, classes)
        base = {"start": start.isoformat(), "end": end.isoformat(), "n": int(cc.sum()),
                "coverage": float(1 - cc[-1] / cc.sum()), "class_counts": cc.tolist(), "classes": classes + [OUTSIDE]}
        bpath.write_text(json.dumps(base, indent=1))
        print(f"reference window {start}..{end}: {base['n']} complaints")

    done = []
    for day in days:
        r = analyse(day.isoformat(), classes, base)
        write_day(r, classes)
        hist.append(hist_row(r))
        done.append(r)
        print(f"{r['date']}: {r['total']} complaints, coverage {pct(r['coverage'])}, PSI {r['psi']:.3f}, {r['status']}")

    hist = save_history(hist)
    has_chart = chart(hist, base["coverage"])
    write_readme(hist, base, has_chart, run_day.isoformat())

    if not done:
        msg = f"Daily run {run_day}: nothing new to check"
    elif len(done) == 1:
        r = done[0]
        msg = (f"Complaints received {r['date']}: {r['total']:,}, coverage {pct(r['coverage'])}, "
               f"label PSI {r['psi']:.3f}, {r['status']}")
    else:
        msg = f"Checked complaints received {done[0]['date']} to {done[-1]['date']} ({len(done)} days)"
    (ROOT / ".commit_msg").write_text(msg)
    print(msg)


if __name__ == "__main__":
    sys.exit(main())
