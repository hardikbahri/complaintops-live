import json, requests
B = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
H = {"User-Agent": "complaintops-live monitor"}
for q in ["size=3&no_aggs=true&sort=created_date_desc",
          "size=3&no_aggs=true&sort=created_date_desc&has_narrative=true",
          "size=3&no_aggs=true&has_narrative=true&date_received_min=2026-06-01",
          "size=3&no_aggs=true&search_term=bank&field=complaint_what_happened",
          "size=3&no_aggs=true&has_narrative=false",
          "size=3&no_aggs=true&has_narrative=true&format=json"]:
    print("===", q)
    try:
        r = requests.get(B + "?" + q, headers=H, timeout=90); print("HTTP", r.status_code, len(r.content))
        d = r.json()
    except Exception as e:
        print("ERR", e); continue
    if isinstance(d, list):
        print("LIST", len(d), json.dumps(d[:1])[:600]); continue
    h = d.get("hits", {}); print("total", h.get("total")); print("meta", json.dumps(d.get("_meta"))[:500])
    for x in h.get("hits", [])[:3]:
        s = x["_source"]; print(s.get("complaint_id"), s.get("date_received"), sorted(s.keys()))
        print("  narrative:", (s.get("complaint_what_happened") or "")[:150])
