"""
model_decision_tree.py  -  Decision Tree classifier + MLflow hyperparameter search
==================================================================================
Supervised task: predict 'Outcome' (diabetes yes/no) from the other columns.
A decision tree learns a flowchart of if/else questions on the features.

It uses the SAME train/test split (test_size=0.2, random_state=42) and the SAME
3-fold cross-validation as the Random Forest scripts, so the accuracy numbers
are directly comparable with model_grid_search.py / model_random_search.py.

Best model = highest cv_accuracy (we pick on cross-validation, NOT on the test set,
so the test set stays an honest, unseen check).

Run:  python model_decision_tree.py
"""

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

from mlflow_utils import search, load_diabetes_data, print_best

data = load_diabetes_data("diabetes_dataset.csv")
features = data.drop(columns="Outcome")

# Same split as the Random Forest scripts
A_tr, A_te, b_tr, b_te = train_test_split(
    features, data["Outcome"], test_size=0.2, random_state=42)


def tree_trial(p):
    """Train ONE decision tree using the hyperparameters in dict p."""
    m = DecisionTreeClassifier(criterion=p["criterion"],              # split quality measure
                               max_depth=p["max_depth"],              # how deep the tree can grow
                               min_samples_split=p["min_samples_split"],
                               random_state=42)

    # 3-fold cross-validation on the TRAINING data (used to choose the best model)
    cv_acc = cross_val_score(m, A_tr, b_tr, cv=3).mean()

    # Fit on all training data, then evaluate on the unseen test data
    m.fit(A_tr, b_tr)
    pred = m.predict(A_te)
    return {"cv_accuracy": cv_acc,
            "test_accuracy": accuracy_score(b_te, pred),
            "precision": precision_score(b_te, pred),   # of predicted diabetic, how many truly are
            "recall": recall_score(b_te, pred),         # of truly diabetic, how many we caught
            "f1": f1_score(b_te, pred)}, m              # keep model for plotting


# 5 depths x 2 criteria x 3 split sizes = 30 trials
#   max_depth None   : tree grows until leaves are pure (usually overfits)
#   gini / entropy   : two ways to measure how "mixed" a node is
grid = {"max_depth": [2, 3, 5, 8, None],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 5, 10]}

if __name__ == "__main__":
    dt = search("DecisionTree", grid, tree_trial, "cv_accuracy", high=True,
                tags={"family": "classification", "unit": "Unit 5", "algorithm": "DecisionTree"})
    print_best("Decision Tree", dt)

    # ---- Draw the best tree (top 3 levels so the picture stays readable) ----
    best = dt.loc[0, "extra"]
    plt.figure(figsize=(16, 8))
    plot_tree(best, feature_names=list(features.columns), class_names=["No", "Yes"],
              filled=True, max_depth=3, fontsize=8)
    plt.title("Best Decision Tree (first 3 levels)")
    plt.savefig("decision_tree_best.png", dpi=120, bbox_inches="tight")
    plt.show()

    # Which features did the tree rely on most?
    print("\nFeature importance:")
    for name, imp in sorted(zip(features.columns, best.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {name:28s} {imp:.3f}")
