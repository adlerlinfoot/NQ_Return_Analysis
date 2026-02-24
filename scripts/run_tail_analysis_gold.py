import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

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
    m = forward_weighted_mean(x)
    return np.sqrt(np.average((x - m)**2, weights=w))

def forward_weighted_skew(x):
    w = np.arange(1, len(x) + 1)
    m = forward_weighted_mean(x)
    s = forward_weighted_std(x)
    return np.average(((x - m) / s)**3, weights=w)

def forward_weighted_kurtosis(x):
    w = np.arange(1, len(x) + 1)
    m = forward_weighted_mean(x)
    s = forward_weighted_std(x)
    return np.average(((x - m) / s)**4, weights=w) - 3

# -----------------------------
# Load NQ + GC
# -----------------------------

nq = pd.read_csv(DATA / "nq_clean.csv")
gc = pd.read_csv(DATA / "gc_clean.csv")

# Merge on date
df = pd.merge(nq, gc, on="date", how="inner")

# -----------------------------
# Compute gold-denominated returns
# -----------------------------

ratio = df["close_nq"] / df["close_gc"]
gold_returns = ratio.pct_change().fillna(0).values
gold_returns = gold_returns[np.isfinite(gold_returns)]

# Gaussian reference
mu = np.mean(gold_returns)
sigma = np.std(gold_returns)
gaussian_sample = np.random.normal(mu, sigma, size=len(gold_returns))

# Forward-weighted version
fw = gold_returns * np.linspace(0.5, 1.5, len(gold_returns))

# -----------------------------
# Compute stats
# -----------------------------

stats_dict = {
    "mean_regular": np.mean(gold_returns),
    "std_regular": np.std(gold_returns),
    "skew_regular": stats.skew(gold_returns),
    "kurtosis_regular": stats.kurtosis(gold_returns),

    "mean_forward": forward_weighted_mean(gold_returns),
    "std_forward": forward_weighted_std(gold_returns),
    "skew_forward": forward_weighted_skew(gold_returns),
    "kurtosis_forward": forward_weighted_kurtosis(gold_returns),
}

with open(OUT / "gold_tail_stats.txt", "w") as f:
    for k, v in stats_dict.items():
        f.write(f"{k}: {v}\n")

# -----------------------------
# FIGURE 1: KDE
# -----------------------------

plt.figure(figsize=(10,6))
sns.kdeplot(gold_returns, label="Regular", linewidth=2)
sns.kdeplot(fw, label="Forward Weighted", linewidth=2)
sns.kdeplot(gaussian_sample, label="Gaussian", linewidth=2)

plt.title("Gold-Denominated NQ Returns – KDE")
plt.xlabel("Return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "gold_kde.png")
plt.close()

# -----------------------------
# FIGURE 2: Tail Exceedance
# -----------------------------

thresholds = np.linspace(0, 5*sigma, 200)

regular_exceed = [np.mean(np.abs(gold_returns) > t) for t in thresholds]
forward_exceed = [np.mean(np.abs(fw) > t) for t in thresholds]
gaussian_exceed = [np.mean(np.abs(gaussian_sample) > t) for t in thresholds]

plt.figure(figsize=(10,6))
plt.plot(thresholds, regular_exceed, label="Regular", linewidth=2)
plt.plot(thresholds, forward_exceed, label="Forward Weighted", linewidth=2)
plt.plot(thresholds, gaussian_exceed, label="Gaussian", linewidth=2)

plt.yscale("log")
plt.title("Gold-Denominated Tail Exceedance – |Return| > Threshold")
plt.xlabel("Threshold")
plt.ylabel("P(|r| > threshold)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "gold_tail_exceedance.png")
plt.close()

print("Gold-denominated tail analysis complete.")