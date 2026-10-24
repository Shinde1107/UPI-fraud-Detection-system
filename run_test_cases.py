"""Run the manual test cases from tests/test_cases.csv and fill in Actual Result / Status."""
import csv, json
from fraud_model import train, predict

model = train()
rows = list(csv.DictReader(open("tests/test_cases.csv")))
for r in rows:
    out = predict(model, json.loads(r["input"]))
    actual = out["message"] if out["status"] == "error" else out["prediction"]
    r["actual_result"] = actual
    r["status"] = "Pass" if r["expected_result"] == actual else "Fail"
with open("tests/test_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("\nPass:", sum(r["status"] == "Pass" for r in rows), "| Fail:", sum(r["status"] == "Fail" for r in rows))
