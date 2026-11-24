# ComplaintOps Live

A daily check on whether the world my [ComplaintOps](https://github.com/hardikbahri/complaintops) complaint-triage model was trained on still looks like the world it would serve.

ComplaintOps classifies US consumer financial complaints into 24 product + issue classes (DistilBERT + LoRA, int8 ONNX, [on Hugging Face](https://huggingface.co/hb-007/complaintops-distilbert-lora-onnx)). Background in the article: [My LLM was confident 99% of the time. It was right 39% of the time.](https://medium.com/@hardik.bahri/my-llm-was-confident-99-of-the-time-it-was-right-39-of-the-time-84c6fc2e34f3)

Every morning a GitHub Actions job:

1. counts the complaints the CFPB received 15 days earlier (late publications have landed by then), by product and issue, from the [public complaint API](https://cfpb.github.io/api/ccdb/),
2. maps every product + issue onto the model's 24 classes, with the same mapping used in training and the class list read from the model's `classes.json`,
3. measures **coverage** (share of complaints the model has a class for) and **label drift** (PSI of the class mix against a 28-day reference window),
4. commits the day's counts, a short report and this dashboard.

Why labels and not predictions: the CFPB has stopped publishing complaint narratives in its public data, so there is no new text to run the model on. The product and issue the consumer picks are still published, and they are exactly what the model predicts. A drop in coverage means new kinds of complaints the model can't name; a rising PSI means the class mix (and the priors behind the model's calibration offsets) has moved, which is the signal to re-check thresholds or retrain.

Reports below are generated automatically by the daily job.

## Dashboard

<!-- LIVE:START -->
First run pending.
<!-- LIVE:END -->

## Status rules

| Status | Meaning |
| --- | --- |
| OK | PSI below 0.10 and coverage within 5 points of the reference window |
| WATCH | PSI 0.10 to 0.25, or coverage 5 to 10 points below reference |
| ALERT | PSI above 0.25, or coverage more than 10 points below reference |
| LOW VOLUME | fewer than 200 complaints that day |

## Layout

- `monitor/run.py`: the daily job
- `monitor/labels.py`: maps CFPB product and issue to the model's classes (same mapping as training)
- `data/daily/`: one CSV per receipt date (product, issue, model class, complaint count)
- `reports/`: one markdown report per receipt date, plus `trend.png`
- `history.csv`: one row per receipt date
- `state/baseline.json`: the frozen reference window

Run it yourself: `pip install -r requirements.txt && python -m monitor.run`
