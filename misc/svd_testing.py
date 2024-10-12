"""Script for experimenting with SVD and PCA."""
# %%
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the covariance matrix
cov_matrix = np.array([[1, 0.7], [0.7, 1]])

# Step 2: Generate random samples based on the covariance matrix
# To do this, we first need to get the mean (center it at origin)
mean = np.zeros(2)  # Mean vector for 3D data
num_samples = 100  # Number of samples to generate

# Generate data points using multivariate normal distribution
data = np.random.multivariate_normal(mean, cov_matrix, num_samples)

# Step 3: Center the data
data_centered = data - np.mean(data, axis=0)

# Step 4: Perform SVD
U, Sigma, Vt = np.linalg.svd(data_centered, full_matrices=False)

# Step 5: Select the top 2 right singular vectors
k = 1
V_k = Vt.T[:, :k]  # Transpose Vt to get (3, 2)

# Step 6a: Project the centered data onto the lower-dimensional space
data_reduced_v = data_centered @ V_k  # Shape will be (num_samples, k)

# Select the top 2 left singular vectors and scale by the singular values
U_k = U[:, :k]  # Take the first 2 columns of U (shape will be (num_samples, 2))
Sigma_k = np.diag(Sigma[:k])  # Top k singular values (shape will be (2, 2))

# Step 6b: Project the centered data onto the lower-dimensional space using U and scaled by Sigma
data_reduced_u = U_k @ Sigma_k  # Shape will be (num_samples, k)

# Get x0
x0 = data[0, 0], data[0, 1]

# First principal component
pc1 = V_k[:, 0]  # Plot

# Project data onto lower dimensional subspace
projected_data = data @ pc1
# Get coordinates of x0 wrt. v1 (in reduced space)
x0_coords_in_reduced_space = projected_data[0]  # coordinate in reduced space
projection = x0_coords_in_reduced_space * pc1  # Datapoint projected onto principle axis

xvals = np.linspace(-3, 3, 100)
princ_axis1 = xvals * (V_k[1] / V_k[0])

plt.figure(figsize=(5, 5))

# Plot the data
plt.plot(data[:, 0], data[:, 1], "*")

# plot the data point x0
plt.plot(x0[0], x0[1], "*", color="r")
plt.plot(xvals, princ_axis1, linewidth=2, color="g")

# Plot the first principal component
plt.plot([0, pc1[0]], [0, pc1[1]], linewidth=2, color="k")
plt.annotate(
    "", xy=pc1, xytext=[0, 0], arrowprops={"arrowstyle": "->", "color": "k", "lw": 3}
)

# Plot the projection
plt.plot(projection[0], projection[1], "m*", markersize=10)
plt.axhline(0, color="black", linewidth=0.5, ls="--")  # X-axis
plt.axvline(0, color="black", linewidth=0.5, ls="--")  # Y-axis
plt.plot(
    [x0[0], projection[0]], [x0[1], projection[1]], "--m"
)
plt.legend(["data", "x0", "axis of v1", "v1", "projection: x1 * v1"])

# plt.annotate('', xy=[0, 0], xytext=[1, 1],
#              arrowprops=dict(arrowstyle='->', color='blue', lw=2))

# %%
plt.plot(data_reduced_v[:, 0], data_reduced_v[:, 1], "*")
plt.plot(data_reduced_u[:, 0], data_reduced_u[:, 1], "x")
plt.title("Scatter Plot coordinates of the 2D representation data")

# %%

# Generate data points using multivariate normal distribution
data = np.random.multivariate_normal(mean, cov_matrix, num_samples)

# Step 3: Center the data
data_centered = data - np.mean(data, axis=0)

# Step 4: Perform SVD
U, Sigma, Vt = np.linalg.svd(data_centered, full_matrices=False)

# Step 5: Select the top 2 left singular vectors and scale by the singular values
k = 2
U_k = U[:, :k]  # Take the first 2 columns of U (shape will be (num_samples, 2))
Sigma_k = np.diag(Sigma[:k])  # Top k singular values (shape will be (2, 2))

# Step 6: Project the centered data onto the lower-dimensional space using U and scaled by Sigma
data_reduced_v = U_k @ Sigma_k  # Shape will be (num_samples, k)

# Display results
print("Original Data Shape:", data.shape)
print("Reduced Data Shape:", data_reduced_v.shape)
print("First 5 Reduced Data Points:\n", data_reduced_v[:5])

plt.figure(figsize=(3, 3))
plt.plot(data_reduced_v[:, 0], data_reduced_v[:, 1], "*")
plt.title("Scatter Plot coordinates of the 2D representation data")
