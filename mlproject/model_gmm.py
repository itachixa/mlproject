"""
model_gmm.py  -  Gaussian Mixture Model (GMM) + MLflow hyperparameter search
============================================================================
GMM is a "soft" clustering method: each cluster is a Gaussian distribution.
We search the number of components and the covariance shape.

Run:  python model_gmm.py
"""

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from mlflow_utils import search, load_scaled_features, print_best

data, X = load_scaled_features("diabetes_dataset.csv")


def scores(labels):
    """Same internal clustering metrics as K-Means (see model_kmeans.py)."""
    return {"silhouette_score": silhouette_score(X, labels),
            "davies_bouldin_index": davies_bouldin_score(X, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X, labels)}


def gmm(p):
    """Train ONE GMM using the hyperparameters in dict p."""
    m = GaussianMixture(p["n_components"],                 # number of Gaussians
                        covariance_type=p["covariance_type"],  # shape of each Gaussian
                        random_state=42)
    labels = m.fit_predict(X)
    # BIC (Bayesian Information Criterion) is an extra metric for GMM:
    # LOWER BIC = better fit with fewer parameters. We log it for reference.
    return {**scores(labels), "bic": m.bic(X)}, None


# 5 component counts x 4 covariance types = 20 trials
#   full      : each cluster has its own free covariance matrix
#   tied      : all clusters share one covariance matrix
#   diag      : each cluster has a diagonal covariance
#   spherical : each cluster has a single variance value
grid = {"n_components": [2, 3, 4, 5, 6],
        "covariance_type": ["full", "tied", "diag", "spherical"]}

if __name__ == "__main__":
    gm = search("GMM", grid, gmm, "silhouette_score", high=True)
    print_best("GMM", gm)
