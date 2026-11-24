# ComplaintOps Live

A daily check of how my [ComplaintOps](https://github.com/hardikbahri/complaintops) complaint-triage model holds up on brand-new, real consumer complaints.

Every morning a GitHub Actions job:

1. pulls the newest complaints published by the US Consumer Financial Protection Bureau ([public API](https://cfpb.github.io/api/ccdb/)),
2. classifies the ones it hasn't seen before with the ComplaintOps model (DistilBERT + LoRA, int8 ONNX, [on Hugging Face](https://huggingface.co/hb-007/complaintops-distilbert-lora-onnx)),
3. compares each prediction with the product and issue the consumer selected,
4. checks for drift (PSI on the predicted class mix and on confidence) against a baseline frozen from the first live runs,
5. commits the day's predictions, a short report and this dashboard.

The question it answers: does the confidence gate (auto-answer at confidence 0.80 or above, route the rest to a person) still deliver the accuracy it showed offline, on complaints the model has never seen? Background in the article: [My LLM was confident 99% of the time. It was right 39% of the time.](https://medium.com/@hardik.bahri/my-llm-was-confident-99-of-the-time-it-was-right-39-of-the-time-84c6fc2e34f3)

Reports below are generated automatically by the daily job.

## Dashboard

<!-- LIVE:START -->
First run pending.
<!-- LIVE:END -->

## Status rules

| Status | Meaning |
| --- | --- |
| OK | PSI below 0.10 and auto-answer accuracy within 4 points of offline |
| WATCH | PSI 0.10 to 0.25, or accuracy 4 to 8 points below offline |
| ALERT | PSI above 0.25, or accuracy more than 8 points below offline |
| LOW VOLUME | fewer than 30 new complaints that day |

## Layout

- `monitor/run.py`: the daily job
- `monitor/labels.py`: maps CFPB product and issue to the model's 24 classes (same mapping as training)
- `data/daily/`: one CSV per run (complaint id, true label, prediction, confidence). No complaint text is stored.
- `reports/`: one markdown report per run, plus `trend.png`
- `history.csv`: one row per run
- `state/`: complaint ids already processed and the frozen drift baseline

Run it yourself: `pip install -r requirements.txt && python -m monitor.run`
