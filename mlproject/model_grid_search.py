"""
model_grid_search.py  -  GridSearchCV on a Random Forest + MLflow tracking
=========================================================================
Grid search tries EVERY combination of the hyperparameters below using
3-fold cross-validation, and keeps the best one.

Run:  python model_grid_search.py
"""

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from mlflow_utils import setup_mlflow, load_diabetes_data

setup_mlflow()
data = load_diabetes_data("diabetes_dataset.csv")

# Supervised task: predict Outcome (diabetes yes/no) from the other columns
A_tr, A_te, b_tr, b_te = train_test_split(
    data.drop(columns="Outcome"), data["Outcome"], test_size=0.2, random_state=42)

# Parameter grid: 3 x 3 x 2 = 18 combinations, each trained 3 times (cv=3)
param_grid = {"n_estimators": [50, 100, 200],     # number of trees
              "max_depth": [3, 5, 8],             # max tree depth
              "min_samples_split": [2, 5]}        # min samples needed to split a node

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)

if __name__ == "__main__":
    with mlflow.start_run(run_name="GridSearch_Optimization"):
        mlflow.set_tag("model", "GridSearch")
        mlflow.set_tag("level", "parent")
        mlflow.set_tag("best", "yes")

        grid.fit(A_tr, b_tr)                                   # run the whole search
        mlflow.log_params(grid.best_params_)                   # best hyperparameters
        mlflow.log_metrics({
            "cv_accuracy": grid.best_score_,                   # mean CV accuracy (train data)
            "test_accuracy": grid.score(A_te, b_te)})          # accuracy on unseen test data

    print("GridSearch best params:", grid.best_params_)
    print("CV accuracy  :", round(grid.best_score_, 4))
    print("Test accuracy:", round(grid.score(A_te, b_te), 4))
