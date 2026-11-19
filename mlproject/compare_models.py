"""
compare_models.py  -  Compare the BEST run of every model, using MLflow tags
============================================================================
Run this AFTER you have run all the model scripts.

What it does:
  1. Finds every run tagged  best = yes  (one per model, set by the model scripts)
  2. Groups them by the tag  family  (clustering / classification / ...)
  3. Ranks models INSIDE each family using that family's metric
  4. Tags the winner of each family with  family_winner = yes
  5. Saves a summary table (model_comparison.csv) and logs it as an MLflow run

WHY "by family"?  Models can only be compared fairly if they are scored with the
same metric on the same task:
   clustering      : K-Means vs GMM vs Hierarchical      -> silhouette score
   classification  : DecisionTree vs RF Grid vs RF Random -> cv_accuracy / test_accuracy
   sequence_hmm / time_series_regression have ONE model each (different tasks and
   metrics: BIC and RMSE), so they cannot be ranked against the others.

Run:  python compare_models.py
"""

import mlflow
import pandas as pd

from mlflow_utils import setup_mlflow, EXPERIMENT_NAME

# family -> (metric used to rank, higher_is_better?, extra metrics to show)
FAMILY_RULES = {
    "clustering": ("silhouette_score", True,
                   ["davies_bouldin_index", "calinski_harabasz_score"]),
    "classification": ("cv_accuracy", True, ["test_accuracy", "f1"]),
    "sequence_hmm": ("bic", False, ["log_likelihood"]),
    "time_series_regression": ("rmse", False, ["mae"]),
}

client = setup_mlflow()
exp = client.get_experiment_by_name(EXPERIMENT_NAME)

# Every run that the model scripts marked as the best of its model
runs = client.search_runs([exp.experiment_id], filter_string="tags.best = 'yes'")

rows = []
for r in runs:
    rows.append({"family": r.data.tags.get("family", "?"),
                 "model": r.data.tags.get("model", "?"),
                 "unit": r.data.tags.get("unit", "?"),
                 "run_id": r.info.run_id,
                 **r.data.metrics})
df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("No runs tagged best=yes found. Run the model scripts first.")

summary = []
for family, (metric, high, extras) in FAMILY_RULES.items():
    part = df[df["family"] == family]
    if part.empty or metric not in part:
        continue
    part = part.sort_values(metric, ascending=not high).reset_index(drop=True)
    cols = ["model", metric] + [c for c in extras if c in part]
    print(f"\n=== {family}  (ranked by {metric}, {'higher' if high else 'lower'} is better) ===")
    print(part[cols].to_string(index=False))

    # Tag the winner so you can filter  tags.family_winner = 'yes'  in the UI
    client.set_tag(part.loc[0, "run_id"], "family_winner", "yes")
    summary.append({"family": family, "winner": part.loc[0, "model"],
                    "metric": metric, "value": part.loc[0, metric],
                    "models_compared": len(part)})

out = pd.DataFrame(summary)
print("\n=== Winner per family ===")
print(out.to_string(index=False))
out.to_csv("model_comparison.csv", index=False)

# Log the comparison itself as a run so it is stored next to the models
with mlflow.start_run(run_name="Model_Comparison"):
    mlflow.set_tags({"level": "summary", "family": "comparison"})
    try:
        mlflow.log_artifact("model_comparison.csv")
    except Exception as e:  # artifact store may be unreachable; table is still saved locally
        print("Could not upload artifact:", e)
