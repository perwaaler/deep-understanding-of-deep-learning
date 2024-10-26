"""Functions for simulating data"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_data(n_samples=200, plot_data=False):
    """2D data with associated binary labels. The function randomizes which
    coordinate is the informative one, making it good for testing attention
    gates and transformers."""
    # Step 1: Simulate class label C with probability 0.5
    C = np.random.binomial(1, 0.5, n_samples)

    # Step 2: Simulate whether x1 or x2 is informative (M)
    M = np.random.binomial(1, 0.5, n_samples)

    # Step 3 & 4: Set informative distributions based on C
    sigma_e = 0.25
    informative_dist1 = np.where(
        C == 1,
        np.random.normal(0, sigma_e, n_samples),
        np.random.normal(1, sigma_e, n_samples),
    )
    informative_dist2 = np.where(
        C == 1,
        np.random.normal(2, sigma_e, n_samples),
        np.random.normal(1, sigma_e, n_samples),
    )

    # Step 5: Simulate the value of x1
    x1 = M * informative_dist1 + (1 - M) * np.random.normal(2, sigma_e, n_samples)

    # Step 6: Simulate the value of x2
    x2 = M * np.random.normal(0, sigma_e, n_samples) + (1 - M) * informative_dist2

    if plot_data:
        xx = np.linspace(-1, 3, 100)
        plt.plot(x1[C == 1], x2[C == 1], "r*")
        plt.plot(x1[C == 0], x2[C == 0], "g*")
        plt.plot(xx, 2 - xx)

    return x1, x2, C


