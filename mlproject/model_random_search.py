"""
model_random_search.py  -  RandomizedSearchCV on a Random Forest + MLflow tracking
=================================================================================
Instead of trying ALL combinations, random search samples n_iter=10 random
combinations from the given ranges/distributions. It is faster and works well
when the search space is large.

Run:  python model_random_search.py
"""

import mlflow
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from mlflow_utils import setup_mlflow, load_diabetes_data

setup_mlflow()
data = load_diabetes_data("diabetes_dataset.csv")

A_tr, A_te, b_tr, b_te = train_test_split(
    data.drop(columns="Outcome"), data["Outcome"], test_size=0.2, random_state=42)

# Distributions: randint(a, b) picks a random integer in [a, b) each time
param_dist = {"n_estimators": randint(50, 300),
              "max_depth": [3, 5, 8, None],        # None = unlimited depth
              "min_samples_split": randint(2, 10)}

rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist,
                          n_iter=10,               # only 10 random combinations
                          cv=3, random_state=42)

if __name__ == "__main__":
    with mlflow.start_run(run_name="RandomSearch_Optimization"):
        mlflow.set_tag("model", "RandomSearch")
        mlflow.set_tag("level", "parent")
        mlflow.set_tag("best", "yes")

        rand.fit(A_tr, b_tr)
        mlflow.log_params(rand.best_params_)
        mlflow.log_metrics({"cv_accuracy": rand.best_score_,
                            "test_accuracy": rand.score(A_te, b_te)})

    print("RandomSearch best params:", rand.best_params_)
    print("CV accuracy  :", round(rand.best_score_, 4))
    print("Test accuracy:", round(rand.score(A_te, b_te), 4))
