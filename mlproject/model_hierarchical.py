"""
model_hierarchical.py  -  Agglomerative (hierarchical) clustering + MLflow search
=================================================================================
Starts with every point as its own cluster and repeatedly merges the closest
pair until 'n_clusters' remain. 'linkage' decides how cluster distance is measured.

Run:  python model_hierarchical.py
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from mlflow_utils import search, load_scaled_features, print_best

data, X = load_scaled_features("diabetes_dataset.csv")


def scores(labels):
    """Same internal clustering metrics as K-Means (see model_kmeans.py)."""
    return {"silhouette_score": silhouette_score(X, labels),
            "davies_bouldin_index": davies_bouldin_score(X, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X, labels)}


def hierarchical(p):
    """Train ONE hierarchical clustering model using the hyperparameters in p."""
    m = AgglomerativeClustering(p["n_clusters"], linkage=p["linkage"])
    return scores(m.fit_predict(X)), None


# 5 cluster counts x 4 linkage types = 20 trials
#   ward     : minimise variance inside merged clusters
#   complete : distance between the FARTHEST points of two clusters
#   average  : average distance between all point pairs
#   single   : distance between the CLOSEST points (can chain into one big cluster)
grid = {"n_clusters": [2, 3, 4, 5, 6],
        "linkage": ["ward", "complete", "average", "single"]}

if __name__ == "__main__":
    hc = search("Hierarchical", grid, hierarchical, "silhouette_score", high=True)
    print_best("Hierarchical", hc)

    # Sanity check from the notebook: cluster sizes for average linkage, k=2.
    # A very unbalanced result (e.g. [760, 8]) means one tiny outlier cluster,
    # which inflates the silhouette score - worth mentioning in your viva.
    print("\nCluster sizes (average linkage, k=2):",
          np.bincount(AgglomerativeClustering(2, linkage="average").fit_predict(X)))
