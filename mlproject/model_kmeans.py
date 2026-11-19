"""
model_kmeans.py  -  K-Means clustering + MLflow hyperparameter search
=====================================================================
Goal: group patients into clusters (unsupervised) and find the best
      'n_clusters' and 'init' method by comparing clustering quality scores.

Run:  python model_kmeans.py
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from mlflow_utils import search, load_scaled_features, print_best

# Load data; X = standardised features (no Outcome column)
data, X = load_scaled_features("diabetes_dataset.csv")


def scores(labels):
    """
    Internal clustering metrics (no true labels needed):
      silhouette_score        : -1..1, HIGHER is better (well separated clusters)
      davies_bouldin_index    : >=0,   LOWER  is better (compact, distinct clusters)
      calinski_harabasz_score : >=0,   HIGHER is better (between/within variance)
    """
    return {"silhouette_score": silhouette_score(X, labels),
            "davies_bouldin_index": davies_bouldin_score(X, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X, labels)}


def kmeans(p):
    """Train ONE K-Means model using the hyperparameters in dict p."""
    m = KMeans(p["n_clusters"],      # how many clusters (k)
               init=p["init"],       # how starting centres are chosen
               n_init=10,            # repeat 10 times with different starts, keep best
               random_state=42)
    labels = m.fit_predict(X)        # fit and return cluster id for each patient
    return scores(labels), None      # (metrics, extra)


# Hyperparameter grid: 5 values of k  x  2 init methods = 10 trials
grid = {"n_clusters": [2, 3, 4, 5, 6],
        "init": ["k-means++", "random"]}

if __name__ == "__main__":
    # Best model = the one with the highest silhouette score
    km = search("KMeans", grid, kmeans, "silhouette_score", high=True)
    print_best("K-Means", km)
