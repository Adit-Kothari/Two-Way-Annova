#!/usr/bin/env python3
"""
ANCOVA on the UCI Student Performance (Math) Dataset

Research question:
Does weekly study time affect final maths grade (G3), after adjusting
for first-period grade (G1)?

Requirements:
    pip install numpy pandas matplotlib seaborn scipy statsmodels

The script downloads the UCI Student Performance dataset, performs the
ANCOVA analysis, checks assumptions, and generates the notebook's plots.
An internet connection is required to download the dataset.
"""


# # ANCOVA on the UCI Student Performance (Math) Dataset
# 
# ### Research question
# Does a student's weekly *study time* (`studytime`: 1 = <2h, 2 = 2-5h, 3 = 5-10h, 4 = >10h)
# affect the *final maths grade* (`G3`), once the *first-period grade* (`G1`) is taken into account?
# 
# ### Why ANCOVA (and not a one-way ANOVA)?
# When checking for the effect of no. of hours studied on a student's grade, one-way ANOVA cannot take into account each individual student's ability since that also affects their final grades. Using first period grades for each student, to be a measure of their ability, and adjusting for it allows us to determine the isolated effect of weekly study time on a student's final period grade.
# 
# ANCOVA does this by using first period grade as a covariate and adjusting the error terms for the effect of the covariate on the dependent variable, and compares the group's adjusted means, i.e. what each group would score if they all started at the same average first period score.
# 
# 
# | | |
# |---|---|
# | **Independent variable (factor)** | `studytime` (4 levels: A <2h, B 2-5h, C 5-10h, D >10h) |
# | **Covariate (X)** | `G1`, first-period grade (0–20) |
# | **Dependent variable (Y)** | `G3`, final grade (0–20) |
# 
# ### Assumptions checked
# 1. Linearity of the G1–G3 relationship
# 2. Homogeneity of regression slopes (study time × G1 interaction not significant)
# 3. Homogeneity of variance of residuals (Levene's test)
# 4. Normality of residuals (Shapiro-Wilk + Q-Q plot)


# ## 1. Imports

import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

sns.set_theme(style="whitegrid", font_scale=1.05)
print(pd.set_option("display.float_format", "{:.3f}".format))


# ### 2. Load & prepare data
# 
# Students with a grade of *0* did not sit the exam; a 0 is not a real score, so these students are removed.

url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
outer = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read()))
inner = zipfile.ZipFile(io.BytesIO(outer.read("student.zip")))
raw = pd.read_csv(inner.open("student-mat.csv"), sep=";")

df = raw[(raw.G3 > 0) & (raw.G2 > 0)].copy()
print(f"Students in maths course: {len(raw)}  ->  removed (zero grades): {len(raw) - len(df)}  ->  N = {len(df)}")

labels = {1: "A: <2h", 2: "B: 2-5h", 3: "C: 5-10h", 4: "D: >10h"}
group_order = list(labels.values())
df["study"] = pd.Categorical(df.studytime.map(labels), categories=group_order)
N, k = len(df), len(group_order)

print("\nGroup sizes:")
print(df.study.value_counts().reindex(group_order))


# **Interpretation.** 38 students with a zero grade are excluded, leaving **N = 357**. The group sizes are unequal (92, 182, 59, 24) because study time is self-reported rather than assigned; the calculations below therefore use each group's own size $n_i$ in place of $n$, and $N$ in place of $kn$.


# ### 3. Observed values (descriptive statistics)

desc = df.groupby("study", observed=True).agg(
    n=("G3", "size"), X_mean_G1=("G1", "mean"), Y_mean_G3=("G3", "mean"))
x_bar, y_bar = df.G1.mean(), df.G3.mean()
print(desc)
print(f"\nOverall means:  X-bar (G1) = {x_bar:.2f},   Y-bar (G3) = {y_bar:.2f}")


# **Interpretation.** Students who study more have higher final grades (C: 12.56, D: 12.67 vs A: 11.47, B: 11.07), but they also had higher first-period grades (C: 12.39, D: 12.25 vs A: 11.03, B: 10.90). The groups did not start at the same level, hence we are using ANCOVA.


# ### 4. Assumption checks


# #### 4a. Linearity of the G1–G3 relationship

r = df.G1.corr(df.G3)
print(f"Pearson correlation r(G1, G3) = {r:.3f}")


# **Interpretation.** G1 and G3 are strongly and linearly related (r = 0.89), so a straight-line adjustment for G1 is appropriate and G1 is a useful covariate. See also Fig. 1, where the points follow straight lines in every group.


# #### 4b. Homogeneity of regression slopes

slope_model = ols("G3 ~ study * G1", data=df).fit()
slope_tab = sm.stats.anova_lm(slope_model, typ=2)
print(slope_tab)

print("\nSeparate slope of G3 on G1 in each group:")
for g in group_order:
    s = df[df.study == g]
    print(f"  {g:>9}: {np.polyfit(s.G1, s.G3, 1)[0]:.3f}")


# **Interpretation.** The group slopes are very similar (0.90, 0.86, 0.92, 0.94) and the study time × G1 interaction is
# not significant (F(3, 349) = 0.45, p = 0.718). The regression lines can be treated as parallel, so a single common
# slope β is valid and standard ANCOVA can be used.


# #### 4c. Homogeneity of variance of residuals (Levene's test)

ancova = ols("G3 ~ study + G1", data=df).fit()
res = ancova.resid

lev = stats.levene(*[res[df.study == g] for g in group_order], center="median")
print(f"Levene's test: F = {lev.statistic:.2f}, p = {lev.pvalue:.3f}")
print("->", "OK (equal variances)" if lev.pvalue > 0.05 else "Violated (unequal variances)")


# **Interpretation.** F(3, 353) = 0.91, p = 0.435 : the residual variances are equal across the four groups, so the pooled error term used by the F test is appropriate (no robust correction is needed).


# #### 4d. Normality of residuals (Shapiro-Wilk)

sw = stats.shapiro(res)
print(f"Shapiro-Wilk: W = {sw.statistic:.3f}, p = {sw.pvalue:.3f}")
print(f"Skewness = {stats.skew(res):.2f},  excess kurtosis = {stats.kurtosis(res):.2f}")


# **Interpretation.** W = 0.992, p = 0.042. The test is marginally significant, which is expected with N = 357 because Shapiro-Wilk is sensitive to even small deviations with large N. The skewness (0.17) and kurtosis (−0.37) are close to 0
# and the Q-Q plot (Fig. 3) follows the reference line, so the residuals are approximately normal; ANCOVA is robust to
# such small departures.


# #### 4e. Do the groups start at the same level? (covariate balance)

bal = stats.f_oneway(*[df.loc[df.study == g, "G1"] for g in group_order])
print(f"One-way ANOVA of G1 across study-time groups: F = {bal.statistic:.2f}, p = {bal.pvalue:.3f}")


# **Interpretation.** F(3, 353) = 4.16, p = 0.006: the groups differ significantly in their starting grades.
# Any raw difference in G3 could therefore reflect these starting differences, which is why G3 must be adjusted for G1.


# ### 5. Calculations


# #### 5a. Group totals

X, Y = df.G1, df.G3
tot = df.assign(X=X, Y=Y, X2=X**2, Y2=Y**2, XY=X*Y).groupby("study", observed=True).agg(
    n=("Y", "size"), sum_X=("X", "sum"), sum_Y=("Y", "sum"),
    sum_X2=("X2", "sum"), sum_Y2=("Y2", "sum"), sum_XY=("XY", "sum"))
tot.loc["Total"] = tot.sum()
print(tot)


# #### 5b. Sums of squares and sums of products

g = tot.drop("Total")
n_i = g.n
GX, GY = X.sum(), Y.sum()

group_term_Y  = (g.sum_Y ** 2 / n_i).sum()
group_term_X  = (g.sum_X ** 2 / n_i).sum()
group_term_XY = (g.sum_X * g.sum_Y / n_i).sum()

SS_bg   = group_term_Y  - GY ** 2 / N
SS_wg   = (Y ** 2).sum() - group_term_Y
SS_bg_x = group_term_X  - GX ** 2 / N
SS_wg_x = (X ** 2).sum() - group_term_X
SP_bg   = group_term_XY - GX * GY / N
SP_wg   = (X * Y).sum() - group_term_XY

ss_table = pd.DataFrame(
    {"Between groups": [SS_bg, SS_bg_x, SP_bg], "Within groups": [SS_wg, SS_wg_x, SP_wg]},
    index=["SS for Y (G3)", "SS for X (G1)", "SP (G1 x G3)"])
print(ss_table.round(2))


# **Interpretation.**
# - $SS_{bg} = 133.06$ : variation in final grades **between** the four study-time groups.
# - $SS_{wg} = 3575.99$ : variation in final grades **within** groups (student-to-student differences) - error.
# - $SS_{wg(x)} = 3610.45$ and $SP_{wg} = 3192.99$ : G1 varies within groups, and G1 and G3 **move together** strongly
#   (large positive $SP_{wg}$) : shows that students with a higher first-period grade tend to have a higher final grade.


# #### 5c. Common slope β and the variation explained by the covariate

beta = SP_wg / SS_wg_x                 # pooled within-group slope
explained_wg = beta * SP_wg            # = SP_wg**2 / SS_wg_x
print(f"beta = SP_wg / SS_wg(x) = {SP_wg:.2f} / {SS_wg_x:.2f} = {beta:.3f}")
print(f"Variation in G3 within groups explained by G1 = beta x SP_wg = {explained_wg:.2f}")
print(f"(check: statsmodels slope for G1 = {ancova.params['G1']:.3f})")


# **Interpretation.** β = 0.884 : within any study-time group, each extra point in the first-period grade predicts about 0.88 extra pointsin the final grade. Of the 3575.99 units of within-group variation, 2823.79 (79%) is explained
# by G1, so most of what looks like random student-to-student noise is actually predictable from where students started.


# #### 5d. Adjusted sums of squares

total_term = (SP_bg + SP_wg) ** 2 / (SS_bg_x + SS_wg_x)
SS_bg_adj = SS_bg - (total_term - explained_wg)
SS_wg_adj = SS_wg - explained_wg

print(f"SS'_bg = SS_bg - [(SP_bg + SP_wg)^2 / (SS_bg(x) + SS_wg(x)) - SP_wg^2 / SS_wg(x)]")
print(f"       = {SS_bg:.2f} - [{total_term:.2f} - {explained_wg:.2f}] = {SS_bg_adj:.2f}")
print(f"SS'_wg = SS_wg - SP_wg^2 / SS_wg(x) = {SS_wg:.2f} - {explained_wg:.2f} = {SS_wg_adj:.2f}")


# **Interpretation.**
# - Adjusted between-group variation fell from 133.06 to 6.99 The bracketed term (126.08) is the part of the
#   group differences explained by the groups' different starting grades: about 95 of the apparent study-time effect.
# - Adjusted within-group (error) variation fell from 3575.99 to 752.20 because G1 explains most of the
#   student-to-student variation.


# #### 5e. Adjusted means, μ and τ

adj = pd.DataFrame({
    "raw_mean_Y": g.sum_Y / n_i,
    "mean_X": g.sum_X / n_i})
adj["X_i - X_bar"] = adj.mean_X - x_bar
adj["adj_mean_Y"] = adj.raw_mean_Y - beta * adj["X_i - X_bar"]       # Y'_i = Y_i - beta (X_i - X_bar)
mu = adj.adj_mean_Y.mean()                                           # mu = (1/k) sum Y'_i
adj["tau"] = adj.adj_mean_Y - mu                                     # tau_i = Y'_i - mu
print(adj.round(2))
print(f"\nmu = mean of adjusted group means = {mu:.2f}")
print(f"sum of tau_i = {adj.tau.sum():.2f}")


# **Interpretation.** Groups C and D started **above** the average G1 (11.27), so their means are adjusted down
# (12.56 → 11.57 and 12.67 → 11.80); groups A and B started below, so they are adjusted up (11.47 → 11.68 and
# 11.07 → 11.40). The raw means span 1.6 points; the adjusted means span only 0.4. All group effects are small
# (τ between −0.21 and +0.19 around μ = 11.61), and they sum to 0 as required by the constraint.


# ### 6. Hypothesis testing

df_bg, df_wg = k - 1, N - k - 1
MSTR_adj = SS_bg_adj / df_bg
MSE_adj = SS_wg_adj / df_wg
F_ancova = MSTR_adj / MSE_adj
p_ancova = stats.f.sf(F_ancova, df_bg, df_wg)
F_crit = stats.f.ppf(0.95, df_bg, df_wg)

print(f"df = ({df_bg}, {df_wg})")
print(f"MSTR' = {SS_bg_adj:.2f} / {df_bg} = {MSTR_adj:.2f}")
print(f"MSE'  = {SS_wg_adj:.2f} / {df_wg} = {MSE_adj:.2f}")
print(f"F = {MSTR_adj:.2f} / {MSE_adj:.2f} = {F_ancova:.2f},  p = {p_ancova:.3f},  critical F(0.05) = {F_crit:.2f}")
print("->", "Reject H0" if p_ancova < 0.05 else "Fail to reject H0")

F_cov = explained_wg / MSE_adj
print(f"\nCovariate G1: F(1, {df_wg}) = {F_cov:.2f}, p = {stats.f.sf(F_cov, 1, df_wg):.3g}")


# #### 6a. Check with statsmodels + effect sizes

anc_tab = sm.stats.anova_lm(ancova, typ=2)
anc_tab["partial_eta_sq"] = anc_tab.sum_sq / (anc_tab.sum_sq + anc_tab.loc["Residual", "sum_sq"])
anc_tab.loc["Residual", "partial_eta_sq"] = np.nan
print(anc_tab.round(3))


# **Interpretation.** statsmodels reproduces the hand calculation exactly (F = 1.09, p = 0.353). The effect of study time
# is negligible (partial η² = 0.009), while the covariate G1 is a very strong predictor (F = 1321.43, partial η² = 0.79).


# #### 6b. ANOVA vs ANCOVA

MSTR = SS_bg / (k - 1)
MSE = SS_wg / (N - k)
F_anova = MSTR / MSE
p_anova = stats.f.sf(F_anova, k - 1, N - k)

comparison = pd.DataFrame({
    "ANOVA (no covariate)": [SS_bg, SS_wg, f"{k-1}, {N-k}", MSTR, MSE, F_anova, p_anova,
                             "Reject H0" if p_anova < 0.05 else "Fail to reject H0"],
    "ANCOVA (controlling G1)": [SS_bg_adj, SS_wg_adj, f"{df_bg}, {df_wg}", MSTR_adj, MSE_adj, F_ancova, p_ancova,
                                "Reject H0" if p_ancova < 0.05 else "Fail to reject H0"]},
    index=["Between SS", "Error SS", "df", "Between MS (MSTR)", "Error MS (MSE)", "F", "p", "Result (alpha = 0.05)"])
print(comparison.to_string(float_format=lambda v: f"{v:.3f}"))
print(f"\nError MS reduced by {100 * (1 - MSE_adj / MSE):.0f}%")


# **Interpretation.**
# - **ANOVA:** F(3, 353) = 4.38, p = 0.005 → reject H₀. Final grades differ significantly between study-time groups.
# - **ANCOVA:** F(3, 352) = 1.09, p = 0.353 → fail to reject H₀. After controlling for the first-period grade, there is
#   **no significant evidence** that study time affects the final grade.
# - Controlling for G1 made the test **79% more precise** (MSE 10.13 → 2.14), so this null result is not due to low power:
#   the apparent study-time effect is explained by the groups' different starting grades.


# ### 7. Post hoc comparisons (only if H₀ is rejected)

if p_ancova < 0.05:
    print(ancova.t_test_pairwise("study", method="bonferroni").result_frame.round(3))
else:
    print(f"p = {p_ancova:.3f} >= 0.05 -> H0 not rejected, so no post hoc tests are needed.")


# ### 8. Visualizations


# #### Fig 1 — ANCOVA fit: parallel regression lines

palette = sns.color_palette("viridis", k)
fig, ax = plt.subplots(figsize=(7.5, 5))
xs = np.array([X.min(), X.max()])
for grp, c in zip(group_order, palette):
    s = df[df.study == grp]
    ax.scatter(s.G1, s.G3, s=14, color=c, alpha=0.4)
    ax.plot(xs, adj.loc[grp, "adj_mean_Y"] + beta * (xs - x_bar), color=c, lw=2.5,
            label=f"{grp} (n = {len(s)})")
ax.axvline(x_bar, color="grey", ls="--", lw=1, label=f"mean G1 = {x_bar:.2f}")
ax.set_xlabel("First-period grade G1 (covariate)")
ax.set_ylabel("Final grade G3")
ax.set_title(f"ANCOVA: parallel lines with common slope β = {beta:.3f}")
ax.legend(title="Weekly study time")
plt.tight_layout()
print(plt.show())


# **Interpretation.** The four lines share the slope β = 0.884 and lie almost on top of each other: for students with the
# same first-period grade, study time makes almost no difference to the final grade. The adjusted means are the heights
# of the lines at the dashed line (mean G1).


# #### Fig 2 — Raw vs adjusted means

se = ancova.get_prediction(pd.DataFrame({
    "study": pd.Categorical(group_order, categories=group_order), "G1": x_bar})).summary_frame()["mean_se"].values
fig, ax = plt.subplots(figsize=(7.5, 4))
pos = np.arange(k)
ax.plot(pos - 0.1, adj.raw_mean_Y, "o", color="grey", ms=9, label="Raw mean (ANOVA)")
ax.errorbar(pos + 0.1, adj.adj_mean_Y, yerr=1.96 * se, fmt="D", color=palette[0],
            capsize=4, ms=8, label="Adjusted mean (ANCOVA), 95% CI")
ax.axhline(mu, color="grey", ls=":", lw=1, label=f"μ = {mu:.2f}")
ax.set_xticks(pos)
ax.set_xticklabels(group_order)
ax.set_ylabel("Final grade G3")
ax.set_title("Raw vs covariate-adjusted final grades")
ax.legend()
plt.tight_layout()
print(plt.show())


# **Interpretation.** The raw means (grey) differ by up to 1.6 points, but after adjusting for G1 (blue) all four
# adjusted means lie within 0.4 points of each other and close to μ = 11.61, with overlapping confidence intervals.


# #### Fig 3 — Q-Q plot of residuals (normality check)

fig, ax = plt.subplots(figsize=(5.5, 5))
stats.probplot(res, dist="norm", plot=ax)
ax.set_title("Q-Q Plot: ANCOVA residuals")
plt.tight_layout()
print(plt.show())


# **Interpretation.** The residuals follow the reference line closely, with only slight deviation in the tails, supporting
# the normality assumption (Section 4d).


# ### 9. Summary of findings
# 
# - **Assumptions:** linearity (r = 0.89), homogeneity of regression slopes (p = 0.718) and equal variances (p = 0.435)
#   are all satisfied; residuals are approximately normal (W = 0.992; only a trivial departure, expected with N = 357).
# - **The groups started at different levels:** students studying 5–10h and >10h already had higher first-period grades
#   (F = 4.16, p = 0.006).
# - **ANOVA vs ANCOVA:** without the covariate, study time appears significant (F(3, 353) = 4.38, p = 0.005). After
#   controlling for G1, it is not (F(3, 352) = 1.09, p = 0.353, partial η² = 0.009) → **fail to reject H₀**.
# - **Why:** 95% of the between-group variation (126.08 of 133.06) is explained by the groups' different starting grades,
#   and G1 reduces the error variance by 79% (β = 0.884), so the null result is not due to low power.
# - **Adjusted means** (11.68, 11.40, 11.57, 11.80) lie within 0.4 points of μ = 11.61, compared with a 1.6-point spread
#   in the raw means.
# - **Conclusion:** students who study more finish with higher maths grades mainly because they were already doing better
#   by the first grading period; among students with the same first-period grade, weekly study time does not significantly
#   predict the final grade.
# - **Limitations:** study time is self-reported and not randomly assigned (the result is not causal), and G1 is recorded
#   during the course, so the analysis tests progress *after* the first grading period.

