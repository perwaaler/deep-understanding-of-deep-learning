"""Example of calibration curve that compares predicted probabilities with
empirical probabilites on simulated data (logisti regression)."""
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression


def softmax(x):
    return 1 / (1 + np.exp(-x))


# Simulate some data
np.random.seed(42)
n_samples = 1000
x = np.random.randn(n_samples)
noise = np.random.randn(n_samples) * 1.1
logits = x * 3 + noise
probabilities = softmax(logits)
labels = np.random.binomial(1, probabilities)

x = x.reshape((n_samples, 1))
labels = labels.reshape((n_samples, 1))

out = LogisticRegression().fit(x, labels)
probabilities_pred = out.predict_proba(x)[:, 1]


# %%
# Generate binary outcomes based on true probabilities
# Compute calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    labels, probabilities_pred, n_bins=20
)

# Plot the calibration curve
plt.figure(figsize=(8, 6))
plt.plot(
    mean_predicted_value, fraction_of_positives, marker="o", label="Calibration curve"
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")

plt.title("Calibration Plot")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.legend()
plt.grid(True)
plt.show()
# %%
