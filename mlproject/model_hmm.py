


import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm

from mlflow_utils import search, load_timeseries, print_best

# S = scaled readings, lengths = readings per patient (so HMM knows sequence boundaries)
g, S, lengths, sc = load_timeseries()


def hmm_trial(p):
    """Train ONE Gaussian HMM using the hyperparameters in dict p."""
    k = p["n_components"]                      # number of hidden states
    m = hmm.GaussianHMM(n_components=k,
                        covariance_type="diag",   # one variance per state
                        n_iter=p["n_iter"],       # max EM iterations
                        random_state=42)
    m.fit(S, lengths)                          # learn parameters with the EM algorithm
    ll = m.score(S, lengths)                   # log-likelihood of the data

    # BIC = -2*logL + (number of free parameters) * ln(number of samples)
    # Free parameters = start probs (k-1) + transition probs k(k-1) + means & variances 2k
    bic = -2 * ll + ((k - 1) + k * (k - 1) + 2 * k) * np.log(len(S))

    # Return the trained model as `extra` so we can use it for plotting later
    return {"bic": bic, "log_likelihood": ll}, m


# 4 state counts x 3 iteration limits = 12 trials
grid = {"n_components": [2, 3, 4, 5],
        "n_iter": [20, 50, 100]}

if __name__ == "__main__":
    # high=False -> LOWER BIC is better
    hm = search("HMM", grid, hmm_trial, "bic", high=False)
    print_best("HMM", hm)

    # ---- Visualise the best HMM ----
    best_model = hm.loc[0, "extra"]                 # the model stored for the best trial
    states = best_model.predict(S, lengths)         # most likely hidden state per reading
    v = g["value"].values[:300]                     # first 300 raw readings
    plt.figure(figsize=(10, 4))
    plt.plot(v, color="gray")                                       # raw signal
    plt.scatter(range(len(v)), v, c=states[:300], cmap="viridis")   # coloured by state
    plt.title("Values colored by hidden state (best HMM)")
    plt.savefig("hmm_best_states.png", dpi=120, bbox_inches="tight")
    plt.show()

    # Count of each measurement code, from the original notebook's checks
    print("\nReading counts per code:\n", g["code"].value_counts())
