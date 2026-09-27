"""
Two-Way ANOVA Worked Example
--------------------------------
Reproduces the worked example from the UCI Student Performance dataset.

Source:
    student-mat.csv from student+performance.zip

Example:
    Factor A = StudyGroup
        studytime = 1 -> <2h
        studytime = 4 -> >10h

    Factor B = sex
        F, M

    Response = G3 (final mathematics grade)

The worked example takes the FIRST 3 observations from each of the
four cells:

    <2h, F : 14, 8, 6
    <2h, M : 14, 5, 10
    >10h, F: 6, 16, 11
    >10h, M: 20, 12, 15

This script:
1. Loads the original CSV from the ZIP file.
2. Selects the same four cells.
3. Takes n=3 observations per cell.
4. Calculates the correction term.
5. Calculates SSA, SSB, SSAB, SSTotal and SSE manually.
6. Calculates df, MS and F.
7. Calculates F-critical and p-values.
8. Prints the final ANOVA table.

Requirements:
    pip install pandas numpy scipy
"""

from pathlib import Path
import zipfile
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import f


# ============================================================
# 1. FIND AND READ THE DATA
# ============================================================

# Put student+performance.zip in the same folder as this .py file.
SCRIPT_DIR = Path(__file__).resolve().parent
ZIP_FILE = SCRIPT_DIR / "student+performance.zip"

if not ZIP_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {ZIP_FILE.name}.\n"
        "Place student+performance.zip in the same folder as this Python file."
    )

with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir = Path(temp_dir)

    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        z.extractall(temp_dir)

    csv_file = temp_dir / "student-mat.csv"

    # Some copies of student+performance.zip contain another archive
    # called student.zip. Handle both structures automatically.
    if not csv_file.exists():
        nested_zips = list(temp_dir.rglob("student.zip"))
        if nested_zips:
            with zipfile.ZipFile(nested_zips[0], "r") as nested:
                nested.extractall(temp_dir / "nested")
            csv_candidates = list((temp_dir / "nested").rglob("student-mat.csv"))
            if csv_candidates:
                csv_file = csv_candidates[0]

    if not csv_file.exists():
        csv_candidates = list(temp_dir.rglob("student-mat.csv"))
        if csv_candidates:
            csv_file = csv_candidates[0]

    if not csv_file.exists():
        raise FileNotFoundError(
            "student-mat.csv was not found inside the ZIP file "
            "or its nested student.zip archive."
        )

    df = pd.read_csv(csv_file, sep=";")


# ============================================================
# 2. CREATE THE STUDYGROUP VARIABLE
# ============================================================

# In the UCI dataset:
# studytime = 1 -> <2 hours
# studytime = 2 -> 2-5 hours
# studytime = 3 -> 5-10 hours
# studytime = 4 -> >10 hours

df["StudyGroup"] = df["studytime"].map({
    1: "<2h",
    2: "2-5h",
    3: "5-10h",
    4: ">10h"
})


# ============================================================
# 3. SELECT THE SAME 12 OBSERVATIONS USED IN THE EXAMPLE
# ============================================================

# The worked example uses the first 3 observations in each cell.

cells = {
    ("<2h", "F"): [14, 8, 6],
    ("<2h", "M"): [14, 5, 10],
    (">10h", "F"): [6, 16, 11],
    (">10h", "M"): [20, 12, 15],
}

# Verify that these are actually the first 3 observations
# from each corresponding cell in the source dataset.
sample_parts = []

for (study_group, sex), expected_grades in cells.items():
    cell = df[
        (df["StudyGroup"] == study_group) &
        (df["sex"] == sex)
    ].head(3)

    actual_grades = cell["G3"].tolist()

    if actual_grades != expected_grades:
        raise ValueError(
            f"Data mismatch for ({study_group}, {sex}).\n"
            f"Expected: {expected_grades}\n"
            f"Found:    {actual_grades}"
        )

    sample_parts.append(cell[["StudyGroup", "sex", "G3"]])

sample = pd.concat(sample_parts, ignore_index=True)


# ============================================================
# 4. DISPLAY THE SELECTED DATA
# ============================================================

print("\n" + "=" * 70)
print("SELECTED DATA USED IN THE WORKED EXAMPLE")
print("=" * 70)

print(sample.to_string(index=False))

print("\nCell values:")

for (study_group, sex), group in sample.groupby(
    ["StudyGroup", "sex"], sort=False
):
    print(f"{study_group:>5} + {sex}: {group['G3'].tolist()}")


# ============================================================
# 5. DEFINE ANOVA PARAMETERS
# ============================================================

a = 2       # number of levels of Factor A
b = 2       # number of levels of Factor B
n = 3       # observations per cell
N = a * b * n

Y = sample["G3"].to_numpy(dtype=float)

T = Y.sum()

# Grand correction term
CF = T**2 / N


# ============================================================
# 6. CALCULATE TOTALS
# ============================================================

# Factor A totals
A_totals = (
    sample.groupby("StudyGroup")["G3"]
    .sum()
)

# Factor B totals
B_totals = (
    sample.groupby("sex")["G3"]
    .sum()
)

# Cell totals
cell_totals = (
    sample.groupby(["StudyGroup", "sex"])["G3"]
    .sum()
)


print("\n" + "=" * 70)
print("TOTALS")
print("=" * 70)

print("\nFactor A totals:")
print(A_totals)

print("\nFactor B totals:")
print(B_totals)

print("\nCell totals:")
print(cell_totals)


# ============================================================
# 7. SUMS OF SQUARES
# ============================================================

# SSA = (1 / bn) * sum(Ti.^2) - T^2/N
SSA = (A_totals.pow(2).sum() / (b * n)) - CF

# SSB = (1 / an) * sum(T.j^2) - T^2/N
SSB = (B_totals.pow(2).sum() / (a * n)) - CF

# SSAB = (1/n) * sum(Tij^2) - T^2/N - SSA - SSB
SSAB = (cell_totals.pow(2).sum() / n) - CF - SSA - SSB

# SSTotal = sum(Y^2) - T^2/N
SSTotal = np.sum(Y**2) - CF

# SSE = SSTotal - SSA - SSB - SSAB
SSE = SSTotal - SSA - SSB - SSAB


# ============================================================
# 8. DEGREES OF FREEDOM
# ============================================================

df_A = a - 1
df_B = b - 1
df_AB = (a - 1) * (b - 1)
df_E = a * b * (n - 1)
df_Total = N - 1


# ============================================================
# 9. MEAN SQUARES
# ============================================================

MSA = SSA / df_A
MSB = SSB / df_B
MSAB = SSAB / df_AB
MSE = SSE / df_E


# ============================================================
# 10. F STATISTICS
# ============================================================

FA = MSA / MSE
FB = MSB / MSE
FAB = MSAB / MSE


# ============================================================
# 11. F CRITICAL VALUE AND P-VALUES
# ============================================================

alpha = 0.05

F_critical = f.ppf(1 - alpha, 1, df_E)

p_A = f.sf(FA, df_A, df_E)
p_B = f.sf(FB, df_B, df_E)
p_AB = f.sf(FAB, df_AB, df_E)


# ============================================================
# 12. PRINT THE CALCULATIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — GRAND CORRECTION TERM")
print("=" * 70)

print(f"T = {T:.0f}")
print(f"N = {N}")
print(f"T²/N = {T:.0f}²/{N} = {CF:.2f}")


print("\n" + "=" * 70)
print("STEP 2 — SUMS OF SQUARES")
print("=" * 70)

print(
    f"SSA = (57² + 80²)/6 - {CF:.2f} = {SSA:.2f}"
)

print(
    f"SSB = (61² + 76²)/6 - {CF:.2f} = {SSB:.2f}"
)

print(
    f"SSAB = (28² + 29² + 33² + 47²)/3 "
    f"- {CF:.2f} - {SSA:.2f} - {SSB:.2f} = {SSAB:.2f}"
)

print(
    f"SSTotal = sum(Y²) - {CF:.2f} = {SSTotal:.2f}"
)

print(
    f"SSError = {SSTotal:.2f} - {SSA:.2f} - "
    f"{SSB:.2f} - {SSAB:.2f} = {SSE:.2f}"
)


# ============================================================
# 13. ANOVA TABLE
# ============================================================

anova_table = pd.DataFrame({
    "Source": [
        "StudyGroup (A)",
        "sex (B)",
        "Interaction (AB)",
        "Error",
        "Total"
    ],
    "SS": [
        SSA,
        SSB,
        SSAB,
        SSE,
        SSTotal
    ],
    "df": [
        df_A,
        df_B,
        df_AB,
        df_E,
        df_Total
    ],
    "MS": [
        MSA,
        MSB,
        MSAB,
        MSE,
        np.nan
    ],
    "F": [
        FA,
        FB,
        FAB,
        np.nan,
        np.nan
    ],
    "p-value": [
        p_A,
        p_B,
        p_AB,
        np.nan,
        np.nan
    ]
})


print("\n" + "=" * 70)
print("STEP 3 — TWO-WAY ANOVA TABLE")
print("=" * 70)

print(
    anova_table.to_string(
        index=False,
        formatters={
            "SS": lambda x: "" if pd.isna(x) else f"{x:.2f}",
            "MS": lambda x: "" if pd.isna(x) else f"{x:.2f}",
            "F": lambda x: "" if pd.isna(x) else f"{x:.2f}",
            "p-value": lambda x: "" if pd.isna(x) else f"{x:.3f}",
        }
    )
)


# ============================================================
# 14. CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

print(f"F-critical at alpha = {alpha} with df=(1,{df_E}): {F_critical:.2f}")

for name, F_value, p_value in [
    ("StudyGroup", FA, p_A),
    ("sex", FB, p_B),
    ("StudyGroup × sex", FAB, p_AB),
]:
    decision = "Reject H0" if p_value < alpha else "Fail to reject H0"

    print(
        f"{name:18s}: F = {F_value:.2f}, "
        f"p = {p_value:.3f} -> {decision}"
    )

print(
    "\nBecause all three F-values are below F-critical and all "
    "p-values are greater than 0.05, none of the three effects "
    "is statistically significant in this n=3-per-cell worked example."
)

print(
    "\nImportant: this is a deliberately small balanced subsample "
    "for demonstrating the hand calculations. It is not the full "
    "N=395 case-study analysis."
)
