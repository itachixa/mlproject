"""
model_rnn.py  -  Simple RNN written from scratch in NumPy + MLflow search
=========================================================================
Task: predict the NEXT glucose value from the previous 5 readings (regression).
The RNN is implemented manually (no TensorFlow/PyTorch) so you can see how
forward pass and back-propagation through time (BPTT) work.

Metrics: RMSE and MAE in real glucose units (LOWER is better).

Run:  python model_rnn.py
"""

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split

from mlflow_utils import search, load_timeseries, print_best

g, S, lengths, sc = load_timeseries()


class RNN:
    """A single-layer vanilla RNN: many inputs over time -> ONE output value."""

    def __init__(self, h, lr):
        r = np.random.RandomState(42)
        self.h, self.lr = h, lr                  # h = hidden units, lr = learning rate
        self.Wx = r.randn(1, h) * 0.1            # input  -> hidden weights
        self.Wh = r.randn(h, h) * 0.1            # hidden -> hidden weights (the "memory")
        self.Wy = r.randn(h, 1) * 0.1            # hidden -> output weights
        self.bh = np.zeros(h)                    # hidden bias
        self.by = np.zeros(1)                    # output bias

    def forward(self, X):
        """X shape: (batch, time_steps, 1). Returns prediction + all hidden states."""
        hs = [np.zeros((len(X), self.h))]        # h_0 = zeros
        for t in range(X.shape[1]):              # walk through time steps
            # h_t = tanh( x_t*Wx + h_{t-1}*Wh + b )
            hs.append(np.tanh(X[:, t] @ self.Wx + hs[-1] @ self.Wh + self.bh))
        # Prediction uses only the LAST hidden state
        return hs[-1] @ self.Wy + self.by, hs

    def step(self, X, y):
        """One gradient-descent update on a mini-batch. Returns the batch MSE loss."""
        p, hs = self.forward(X)
        d = 2 * (p - y) / len(X)                 # derivative of MSE w.r.t. prediction
        gWy, gby = hs[-1].T @ d, d.sum(0)        # output-layer gradients
        dh = d @ self.Wy.T                       # gradient flowing into last hidden state
        gWx, gWh, gbh = 0, 0, 0
        # ---- Back-propagation through time: go from last step to first ----
        for t in reversed(range(X.shape[1])):
            dz = dh * (1 - hs[t + 1] ** 2)       # tanh derivative
            gWx = gWx + X[:, t].T @ dz
            gWh = gWh + hs[t].T @ dz
            gbh = gbh + dz.sum(0)
            dh = dz @ self.Wh.T                  # pass gradient to previous time step
        # ---- Update weights; clip gradients to [-1, 1] to avoid exploding gradients
        for w, gr in [(self.Wx, gWx), (self.Wh, gWh), (self.bh, gbh),
                      (self.Wy, gWy), (self.by, gby)]:
            w -= self.lr * np.clip(gr, -1, 1)
        return np.mean((p - y) ** 2)

    def fit(self, X, y, epochs):
        """Train for `epochs` passes over data using mini-batches of 64."""
        r = np.random.RandomState(42)
        losses = []
        for _ in range(epochs):
            idx = r.permutation(len(X))          # shuffle each epoch
            losses.append(np.mean([self.step(X[idx[i:i + 64]], y[idx[i:i + 64]])
                                   for i in range(0, len(X), 64)]))
        return losses

    def predict(self, X):
        return self.forward(X)[0]


# ---------------- Build the supervised dataset (sliding windows) ----------------
# For each patient: window of 5 consecutive readings -> the 6th reading is the target.
Xw, yw, s = [], [], 0
for n in lengths:                                # n = number of readings for this patient
    q = S[s:s + n, 0]                            # this patient's scaled readings
    s += n
    for i in range(n - 5):
        Xw.append(q[i:i + 5])                    # input: 5 past values
        yw.append(q[i + 5])                      # target: next value
Xw = np.array(Xw).reshape(-1, 5, 1)              # shape (samples, time_steps, features)
yw = np.array(yw).reshape(-1, 1)

# Use a random subset of 5000 windows to keep NumPy training fast
keep = np.random.RandomState(42).choice(len(Xw), 5000, replace=False)
X_tr, X_te, y_tr, y_te = train_test_split(Xw[keep], yw[keep], test_size=0.2, random_state=42)

scale = sc.scale_[0]                             # std of original data -> back to real units

# Baseline: always predict the training mean. A useful RNN must beat this RMSE.
baseline_rmse = np.sqrt(np.mean((y_te - y_tr.mean()) ** 2)) * scale


def rnn_trial(p):
    """Train ONE RNN with hyperparameters in dict p and evaluate on the test set."""
    m = RNN(p["hidden_units"], p["learning_rate"])
    losses = m.fit(X_tr, y_tr, p["epochs"])
    # Log the loss curve in MLflow (step = epoch) -> see it under the run's Metrics tab
    for e, l in enumerate(losses):
        mlflow.log_metric("train_loss", float(l), step=e)
    err = m.predict(X_te) - y_te
    return {"rmse": np.sqrt(np.mean(err ** 2)) * scale,
            "mae": np.mean(np.abs(err)) * scale}, (m, losses)


# 3 sizes x 2 learning rates x 2 epoch counts = 12 trials
grid = {"hidden_units": [8, 16, 32],
        "learning_rate": [0.01, 0.05],
        "epochs": [20, 40]}

if __name__ == "__main__":
    print("Baseline RMSE (predict the mean):", baseline_rmse)

    # high=False -> LOWER RMSE is better
    rn = search("RNN", grid, rnn_trial, "rmse", high=False)
    print_best("RNN", rn)

    # ---- Plot the best RNN: training loss + predictions vs real values ----
    model, losses = rn.loc[0, "extra"]
    pred = model.predict(X_te) * scale + sc.mean_[0]     # undo scaling -> real units
    real = y_te * scale + sc.mean_[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(losses)
    ax[0].set_title("Best RNN - training loss")
    ax[1].plot(real[:100, 0], label="Real")
    ax[1].plot(pred[:100, 0], label="RNN")
    ax[1].set_title("First 100 test samples")
    ax[1].legend()
    plt.savefig("rnn_best.png", dpi=120, bbox_inches="tight")
    plt.show()
