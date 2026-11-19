"""
mlflow_utils.py
---------------
Shared helpers used by every model script, so we do not repeat the same code.

It contains:
  1. MLflow setup (tracking server + experiment name)
  2. search()            -> generic "grid search with MLflow tracking" function
  3. load_diabetes_data  -> loads the tabular diabetes.csv (used by clustering + RF)
  4. load_timeseries     -> loads the time-series CSV (used by HMM + RNN)

Before running any model script, start the MLflow server in another terminal:
    mlflow ui --port 5000
and open http://127.0.0.1:5000 in your browser to see the results.
"""

import itertools
import logging
import os
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---- Silence noisy warnings so the console output stays readable ----
warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# ---- Fix the random seed so results are reproducible ----
np.random.seed(42)

# ---- MLflow configuration ----
# Where MLflow stores runs. You can override with: set MLFLOW_TRACKING_URI=...
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "Diabetes_Early_Detection_Simple"


def setup_mlflow():
    """Connect to the MLflow server and select (or create) the experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.MlflowClient()


def search(tag, grid, fn, key, high=True, tags=None):
    """
    Try EVERY combination of hyperparameters in `grid` and log each one to MLflow.

    Parameters
    ----------
    tag  : model name (e.g. "KMeans"). Used for run names and tags.
    grid : dict {param_name: [list of values to try]}
    fn   : function(params_dict) -> (metrics_dict, extra)
           It trains ONE model with those params and returns its metrics.
           `extra` can be anything we want to keep (e.g. the trained model).
    key  : name of the metric used to decide which combination is the BEST.
    high : True  -> bigger `key` is better (e.g. silhouette score)
           False -> smaller `key` is better (e.g. BIC, RMSE)
    tags : dict of EXTRA tags (family, unit, algorithm ...) added to the parent
           run and every trial run. These are what let us filter / compare
           models in the MLflow UI and in compare_models.py.

    MLflow structure created:
        parent run  "<tag>_Optimization"        (summary + best params)
          +-- nested child run per combination  (params + metrics of one trial)

    Returns a DataFrame of all trials sorted best-first (row 0 = best).
    """
    client = setup_mlflow()
    rows = []

    # Tags that describe HOW this model should be compared with others
    all_tags = {"model": tag,
                "metric_key": key,                                # metric used to pick the best
                "metric_direction": "higher" if high else "lower",
                **(tags or {})}                                   # family, unit, algorithm...

    # ----- PARENT RUN: groups all trials of this model together -----
    with mlflow.start_run(run_name=tag + "_Optimization"):
        mlflow.set_tags(all_tags)
        mlflow.set_tag("level", "parent")

        # itertools.product builds every combination of the grid values
        for values in itertools.product(*grid.values()):
            p = dict(zip(grid, values))  # e.g. {"n_clusters": 3, "init": "random"}

            # ----- CHILD RUN: one hyperparameter combination -----
            with mlflow.start_run(run_name=tag + "_" + "_".join(map(str, values)),
                                  nested=True) as run:
                mlflow.set_tags(all_tags)
                mlflow.set_tag("level", "trial")
                mlflow.log_params(p)                 # log the hyperparameters
                metrics, extra = fn(p)               # train + evaluate the model
                mlflow.log_metrics(metrics)          # log the evaluation scores
                rows.append({**p, **metrics, "run_id": run.info.run_id, "extra": extra})

        # Sort all trials so the best one is on top
        df = (pd.DataFrame(rows)
                .sort_values(key, ascending=not high)
                .reset_index(drop=True))

        # Record the winning hyperparameters + score on the parent run
        mlflow.log_params({"best_" + k: df.loc[0, k] for k in grid})
        mlflow.log_metric("best_" + key, float(df.loc[0, key]))

    # Mark the best child run with tag best=yes (filter it in the MLflow UI)
    client.set_tag(df.loc[0, "run_id"], "best", "yes")
    return df


def print_best(name, df):
    """Print the best trial (top row) without the bulky helper columns."""
    print(f"\n=== Best {name} configuration ===")
    print(df.drop(columns=["run_id", "extra"]).head(1).to_string(index=False))
    print("\nAll trials (best first):")
    print(df.drop(columns=["run_id", "extra"]).to_string(index=False))


def load_diabetes_data(path="diabetes.csv"):
    """
    Load the tabular diabetes dataset (Pima-style, with an 'Outcome' column).
    Returns:
        data : the full DataFrame
    """
    return pd.read_csv(path)


def load_scaled_features(path="diabetes.csv"):
    """
    Load diabetes.csv and return (data, X) where X = standardised features
    WITHOUT the 'Outcome' label. Clustering is unsupervised, so we do not
    use the label. Scaling (mean 0, std 1) is needed because distance-based
    algorithms are sensitive to feature ranges.
    """
    data = load_diabetes_data(path)
    X = StandardScaler().fit_transform(data.drop(columns="Outcome"))
    return data, X


def load_timeseries(path="diabetes_merged_date-time-sorted-includes-patient-id.csv"):
    """
    Load the glucose / insulin time-series dataset used by HMM and RNN.

    Steps:
      1. Keep only the measurement codes we care about (glucose related).
      2. Sort by patient and time, so each patient's readings are in order.
      3. Drop patients with fewer than 10 readings (too short to model).
      4. Standardise the 'value' column.

    Returns:
      g       : filtered DataFrame
      S       : scaled values, shape (n_samples, 1)
      lengths : list with number of readings per patient (needed by HMM/RNN
                to know where one patient's sequence ends and the next begins)
      sc      : the fitted StandardScaler (to convert back to real units)
    """
    ts = pd.read_csv(path)
    g = ts[ts["code"].isin([48, 57, 58, 59, 60, 61, 62, 63, 64])]
    g = g.sort_values(["patient_id", "datetime"])
    g = g[g.groupby("patient_id")["value"].transform("size") >= 10]
    lengths = g.groupby("patient_id", sort=False).size().tolist()
    sc = StandardScaler()
    S = sc.fit_transform(g[["value"]])
    return g, S, lengths, sc
