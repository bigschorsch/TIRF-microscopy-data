import numpy as np
from scipy import stats

# ==========================
# INPUT: slopes (1/min)
# ==========================
# Enter the well-wise slopes for each temperature condition:
slopes_26 = np.array([
    0.000998, 0.000374, -0.000446, 0.000529
])

slopes_37 = np.array([
   -0.000371, 0.002038, -0.002955, 0.001859
])

alpha = 0.05  # significance level


# ==========================
# Helper: summary + tests
# ==========================
def summarize(x: np.ndarray, name: str):
    x = np.asarray(x, dtype=float)
    n = x.size
    mean = x.mean()
    sd = x.std(ddof=1) 
    sem = sd / np.sqrt(n)
    # 95% CI for mean (t-based)
    if n > 1:
        tcrit = stats.t.ppf(0.975, df=n-1)
        ci = (mean - tcrit * sem, mean + tcrit * sem)
    else:
        ci = (np.nan, np.nan)
    print(f"\n{name}")
    print(f"n={n}, mean={mean:.6g}, SD={sd:.6g}, SEM={sem:.6g}")
    print(f"95% CI for mean: [{ci[0]:.6g}, {ci[1]:.6g}]")

def one_sample_test_vs_zero(x: np.ndarray, name: str):
    x = np.asarray(x, dtype=float)
    n = x.size
    # One-sample t-test against 0
    t_stat, p_two = stats.ttest_1samp(x, popmean=0.0, alternative="two-sided")
    # Effect size: Cohen's d (one-sample)
    sd = x.std(ddof=1)
    d = x.mean() / sd if (n > 1 and sd > 0) else np.nan
    print(f"\nOne-sample t-test vs 0 ({name})")
    print(f"t={t_stat:.6g}, df={n-1}, p(two-sided)={p_two:.6g}, Cohen's d={d:.6g}")
    return t_stat, p_two, d

def two_sample_welch_test(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Welch's t-test (unequal variances)
    t_stat, p_two = stats.ttest_ind(x, y, equal_var=False, alternative="two-sided")
    print("\nWelch's two-sample t-test (26°C vs 37°C)")
    print(f"t={t_stat:.6g}, p(two-sided)={p_two:.6g}")
    return t_stat, p_two

def mann_whitney_u(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Nonparametric alternative (two-sided)
    u_stat, p_two = stats.mannwhitneyu(x, y, alternative="two-sided")
    print("\nMann–Whitney U test (26°C vs 37°C)")
    print(f"U={u_stat:.6g}, p(two-sided)={p_two:.6g}")
    return u_stat, p_two


# ==========================
# Run
# ==========================
summarize(slopes_26, "26°C slopes")
summarize(slopes_37, "37°C slopes")

t26, p26, d26 = one_sample_test_vs_zero(slopes_26, "26°C")
t37, p37, d37 = one_sample_test_vs_zero(slopes_37, "37°C")

t_welch, p_welch = two_sample_welch_test(slopes_26, slopes_37)
u_stat, p_mwu = mann_whitney_u(slopes_26, slopes_37)

print("\nDecision (alpha=0.05)")
print(f"26°C vs 0: {'SIGNIFICANT' if p26 < alpha else 'n.s.'} (p={p26:.6g})")
print(f"37°C vs 0: {'SIGNIFICANT' if p37 < alpha else 'n.s.'} (p={p37:.6g})")
print(f"26°C vs 37°C (Welch): {'SIGNIFICANT' if p_welch < alpha else 'n.s.'} (p={p_welch:.6g})")
print(f"26°C vs 37°C (MWU): {'SIGNIFICANT' if p_mwu < alpha else 'n.s.'} (p={p_mwu:.6g})")
