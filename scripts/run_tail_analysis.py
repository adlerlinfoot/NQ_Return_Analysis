import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import scipy.stats as stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "figures_and_stats"
OUT.mkdir(exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------

def forward_weighted_mean(x):
    w = np.arange(1, len(x) + 1)
    return np.average(x, weights=w)

def forward_weighted_std(x):
    w = np.arange(1, len(x) + 1)
    mean_fw = forward_weighted_mean(x)
    return np.sqrt(np.average((x - mean_fw)**2, weights=w))

def forward_weighted_skew(x):
    w = np.arange(1, len(x) + 1)
    mean_fw = forward_weighted_mean(x)
    std_fw = forward_weighted_std(x)
    return np.average(((x - mean_fw) / std_fw)**3, weights=w)

def forward_weighted_kurtosis(x):
    w = np.arange(1, len(x) + 1)
    mean_fw = forward_weighted_mean(x)
    std_fw = forward_weighted_std(x)
    return np.average(((x - mean_fw) / std_fw)**4, weights=w) - 3

# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(DATA / "nq_clean.csv")
returns = df["return_nq"].values
returns = returns[np.isfinite(returns)]

# Gaussian reference
mu = np.mean(returns)
sigma = np.std(returns)
gaussian_sample = np.random.normal(mu, sigma, size=len(returns))

# -----------------------------
# Compute stats
# -----------------------------

stats_dict = {
    "mean_regular": np.mean(returns),
    "std_regular": np.std(returns),
    "skew_regular": stats.skew(returns),
    "kurtosis_regular": stats.kurtosis(returns),

    "mean_forward": forward_weighted_mean(returns),
    "std_forward": forward_weighted_std(returns),
    "skew_forward": forward_weighted_skew(returns),
    "kurtosis_forward": forward_weighted_kurtosis(returns),
}

# Save stats
with open(OUT / "nq_tail_stats.txt", "w") as f:
    for k, v in stats_dict.items():
        f.write(f"{k}: {v}\n")

# -----------------------------
# FIGURE 1: KDE (histogram without bars)
# -----------------------------

plt.figure(figsize=(10,6))
sns = __import__("seaborn")

sns.kdeplot(returns, label="Regular", linewidth=2)
sns.kdeplot(returns * np.linspace(0.5, 1.5, len(returns)), label="Forward Weighted", linewidth=2)
sns.kdeplot(gaussian_sample, label="Gaussian", linewidth=2)

plt.title("NQ Daily Returns – KDE")
plt.xlabel("Return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "nq_kde.png")
plt.close()

# -----------------------------
# FIGURE 2: Tail Exceedance Plot
# -----------------------------

thresholds = np.linspace(0, 5*sigma, 200)

regular_exceed = [np.mean(np.abs(returns) > t) for t in thresholds]
forward_exceed = [np.mean(np.abs(returns * np.linspace(0.5, 1.5, len(returns))) > t) for t in thresholds]
gaussian_exceed = [np.mean(np.abs(gaussian_sample) > t) for t in thresholds]

plt.figure(figsize=(10,6))
plt.plot(thresholds, regular_exceed, label="Regular", linewidth=2)
plt.plot(thresholds, forward_exceed, label="Forward Weighted", linewidth=2)
plt.plot(thresholds, gaussian_exceed, label="Gaussian", linewidth=2)

plt.yscale("log")
plt.title("Tail Exceedance – |Return| > Threshold")
plt.xlabel("Threshold")
plt.ylabel("P(|r| > threshold)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "nq_tail_exceedance.png")
plt.close()

print("Tail analysis complete. Outputs saved to figures_and_stats/")