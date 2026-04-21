# Python script to demonstrate anomaly detection using statistical methods
# with visualisation

# Import the required modules
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------
# 1. Generate a synthetic dataset
# ---------------------------------------------------------------------
np.random.seed(42)  # For reproducibility

# Normal data
normal_data = np.random.normal(loc=50, scale=5, size=200)

# Inject some anomalies
anomalies = np.array([90, 95, 100, 10, 5])

# Combine the above into a single dataset
data = np.concatenate([normal_data, anomalies])


# ---------------------------------------------------------------------
# 2. Z-score method
# ---------------------------------------------------------------------
def z_score_detection(data, threshold=3):
    # TODO: Document function
    """

    """
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold


# ---------------------------------------------------------------------
# 3. Modified Z-score Method
# ---------------------------------------------------------------------
def modified_z_score_detection(data, threshold=3.5):
    # TODO: Document function
    """
    Detect anomalies using the Modified Z-Score Detection method (based on media).

    More robust to outliers than standard z-score.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))  # Median Absolute Deviation

    modified_z_scores = .6745 * (data - median) / mad
    return np.abs(modified_z_scores) > threshold


# ---------------------------------------------------------------------
# 4. IQR Method
# ---------------------------------------------------------------------
def iqr_detection(data):
    """

    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return (data < lower_bound) | (data > upper_bound)


# ---------------------------------------------------------------------
# 5. Gaussian Probability Method
# ---------------------------------------------------------------------
def gaussian_detection(data, threshold=.01):
    """

    """
    mean = np.mean(data)
    std = np.std(data)

    probabilities = stats.norm.pdf(data, mean, std)
    return probabilities < threshold


# ---------------------------------------------------------------------
# 6. Apply the above methods
# ---------------------------------------------------------------------
z_anomalies = z_score_detection(data)
mod_z_anomalies = modified_z_score_detection(data)
iqr_anomalies = iqr_detection(data)
gaussian_anomalies = gaussian_detection(data)

# ---------------------------------------------------------------------
# 7. Visualisation
# ---------------------------------------------------------------------
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

methods = [
    ("Z-score", z_anomalies),
    ("Modified Z-score", mod_z_anomalies),
    ("IQR", iqr_anomalies),
    ("Gaussian", gaussian_anomalies)
]

for n, (title, mask) in enumerate(methods, 1):
    plt.subplot(2, 2, n)

    # Plot the normal points
    plt.scatter(range(len(data)), data, label="Normal",alpha=0.6)

    # Highlight anomalies
    plt.scatter(
        np.where(mask),
        data[mask],
        color="red",
        label="Anomaly",
        s=80
    )

    plt.title(f"{title} Method")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()

plt.tight_layout()
plt.show()
