import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the true function to optimize (can be any function)
def true_function(X):
    return np.sin(X)

# Define the acquisition function (Expected Improvement)
def expected_improvement(X, gaussian_process, evaluated_loss, xi=0.01):
    mean, std = gaussian_process.predict(X, return_std=True)
    Z = (mean - evaluated_loss - xi) / std
    return (mean - evaluated_loss - xi) * norm.cdf(Z) + std * norm.pdf(Z)

# Generate initial data points (random or manually selected)
X_init = np.array([[-3.], [1.], [2.5]])
y_init = true_function(X_init)

# Define the range of the domain
X = np.linspace(-5, 5, 1000).reshape(-1, 1)

# Fit the GP to the initial data points
kernel = RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True)
gp.fit(X_init, y_init)

# Predict mean and variance with GP
y_pred, sigma = gp.predict(X, return_std=True)

# Calculate expected improvement
ei = expected_improvement(X, gp, np.min(y_init))

# Visualization
plt.figure(figsize=(10, 5))

# Plot GP fit
plt.subplot(1, 2, 1)
plt.plot(X, true_function(X), 'r:', label='True Function')
plt.plot(X, y_pred, 'b-', label='GP Mean')
plt.fill_between(X.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='blue')
plt.scatter(X_init, y_init, color='red', label='Initial Points')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.title('Gaussian Process Fit')
plt.legend()

# Plot Expected Improvement
plt.subplot(1, 2, 2)
plt.plot(X, ei, 'g-', label='Expected Improvement')
plt.xlabel('X')
plt.ylabel('Expected Improvement')
plt.title('Expected Improvement (EI)')
plt.legend()
plt.tight_layout()
plt.savefig('readme_figs/EI.pdf')
plt.show()
