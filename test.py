from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dropwise_metrics.metrics.entropy import PredictiveEntropyMetric

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

metric = PredictiveEntropyMetric(model, tokenizer, task_type="sequence-classification", num_passes=10)
metric.update(["The movie was amazing!", "Terrible acting."])
results = metric.compute()

for res in results:
    print()
    for k, v in res.items():
        print(f"{k:>20}: {v}")
