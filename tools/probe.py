import json, requests
B = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
H = {"User-Agent": "complaintops-live monitor"}
def get(q):
    r = requests.get(B + "?" + q, headers=H, timeout=120); print("HTTP", r.status_code, len(r.content), q); return r.json()
for q in ["size=0&date_received_min=2026-11-22&date_received_max=2026-11-22",
          "size=0&date_received_min=2026-11-22&date_received_max=2026-11-23",
          "size=1&date_received_min=2026-11-01&date_received_max=2026-11-02"]:
    try: d = get(q)
    except Exception as e: print("ERR", e); continue
    print("total", d.get("hits", {}).get("total"), "aggs:", list((d.get("aggregations") or {}).keys()))
a = d["aggregations"]
for k in ["product", "issue"]:
    v = a.get(k, {}); inner = v.get(k, v)
    print(k, "keys", list(v.keys()), "inner keys", list(inner.keys()) if isinstance(inner, dict) else None)
    b = inner.get("buckets", []); print(" n buckets", len(b), "other", inner.get("sum_other_doc_count"))
    for x in b[:3]: print("  ", json.dumps(x)[:400])
d = get("size=0&date_received_min=2026-11-01&date_received_max=2026-11-02&product=" + requests.utils.quote("Debt collection"))
print("filtered total", d["hits"]["total"]); iv = d["aggregations"]["issue"]; iv = iv.get("issue", iv)
print("issue buckets", [(x["key"], x["doc_count"]) for x in iv["buckets"]][:12], "other", iv.get("sum_other_doc_count"))
