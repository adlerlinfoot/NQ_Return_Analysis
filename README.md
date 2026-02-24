# NQ\Return\_Analysis

Return analysis of the Nasdaq 100 E-mini continuous futures contract (NQ).



OVERVIEW



This project investigates the statistical behavior of Nasdaq‑100 (NQ) daily returns under multiple transformations, with a focus on tail risk, distributional shape, and the influence of currency effects. Using both USD‑denominated and gold‑denominated price series, the analysis evaluates standard and forward‑weighted return profiles to identify structural changes in skewness, kurtosis, and tail‑exceedance behavior over time.



KEY FINDINGS

* USD-denominated returns exhibit elevated excess kurtosis and mild negative skew, reflecting persistent left-tail risk and currency amplified shocks.
* Forward-weighted returns show a higher excess kurtosis value compared to standard returns, highlighting a higher frequency of extreme events in recent years.
* Gold-denominated returns display substantially less aggressive return patterns, reinforcing the idea that currency is amplifying market fluctuations.
* A skew inversion in gold-denominated forward-weighted returns points to a structural shift towards more upside-tilted extreme events in the recent regime.



REPRODUCIBILITY



Run "python scripts/data\_cleaner/clean\_daily.py" in terminal to build clean dataset, then run each analysis script individual to produce figures and metrics. All final outputs are routed into figures\_and\_stats.



REQUIREMENTS



Python 3.10+ with standard scientific libraries (NumPy, Pandas, Matplotlib, SciPy, Pathlib, Seaborn).



FULL REPORT 



A concise summary of methods and results is available in:

report/NQ\_return\_analysis.pdf



LICENSE 



MIT

