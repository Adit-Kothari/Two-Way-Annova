# Two-Way-Annova
 # Two-Way ANOVA — Worked Example

This project demonstrates a **Two-Way Analysis of Variance (Two-Way ANOVA)** using a small balanced sample from the **UCI Student Performance dataset**.

The example examines whether:

* **Weekly study time** affects final mathematics grade (`G3`)
* **Sex** affects final mathematics grade (`G3`)
* There is an **interaction between study time and sex**

The Python implementation reproduces the hand calculations presented in the worked example.

---

## Dataset

The analysis uses the **UCI Student Performance — Mathematics dataset (`student-mat.csv`)**.

The original dataset contains **395 students**.

For this worked example, a small balanced subsample is selected:

* 2 levels of StudyGroup
* 2 levels of Sex
* 3 students per combination

Therefore:

$$
a = 2,\quad b = 2,\quad n = 3
$$

and

$$
N = abn = 2\times2\times3 = 12
$$

### Variables

| Variable     | Description             |
| ------------ | ----------------------- |
| `StudyGroup` | Weekly study time       |
| `sex`        | Student sex             |
| `G3`         | Final mathematics grade |

The study-time categories used in this example are:

* `studytime = 1` → `<2h`
* `studytime = 4` → `>10h`

---

## Worked Example Data

The following 12 observations are used:

| StudyGroup | Sex | G3 |
| ---------- | --- | -- |
| `<2h`      | F   | 14 |
| `<2h`      | F   | 8  |
| `<2h`      | F   | 6  |
| `<2h`      | M   | 14 |
| `<2h`      | M   | 5  |
| `<2h`      | M   | 10 |
| `>10h`     | F   | 6  |
| `>10h`     | F   | 16 |
| `>10h`     | F   | 11 |
| `>10h`     | M   | 20 |
| `>10h`     | M   | 12 |
| `>10h`     | M   | 15 |

The cell totals are:

|        |  F |  M |
| ------ | -: | -: |
| `<2h`  | 28 | 29 |
| `>10h` | 33 | 47 |

Therefore:

* Row total for `<2h` = **57**
* Row total for `>10h` = **80**
* Column total for F = **61**
* Column total for M = **76**
* Grand total = **137**

---

# Statistical Model

The Two-Way ANOVA model is:

$$
Y_{ijk}=\mu+\alpha_i+\beta_j+(\alpha\beta)_{ij}+\epsilon_{ijk}
$$

where:

* \(\mu\) = overall mean
* \(\alpha_i\) = effect of StudyGroup
* \(\beta_j\) = effect of sex
* \((\alpha\beta)_{ij}\) = interaction effect
* \(\epsilon_{ijk}\) = random error

---

# Calculations

## 1. Grand Correction Term

The correction term is:

$$
CF=\frac{T^2}{N}
$$

where:

$$
T=137,\qquad N=12
$$

Therefore:

$$
CF=\frac{137^2}{12}=1564.08
$$

---

## 2. Sum of Squares for StudyGroup

$$
SS_A=
\frac{1}{bn}\sum_i T_{i.}^2-\frac{T^2}{N}
$$

Using the row totals:

$$
SS_A=
\frac{57^2+80^2}{6}-1564.08
$$

$$
SS_A=44.08
$$

---

## 3. Sum of Squares for Sex

$$
SS_B=
\frac{1}{an}\sum_j T_{.j}^2-\frac{T^2}{N}
$$

Using the column totals:

$$
SS_B=
\frac{61^2+76^2}{6}-1564.08
$$

$$
SS_B=18.75
$$

---

## 4. Interaction Sum of Squares

$$
SS_{AB}
=
\frac{1}{n}\sum_i\sum_jT_{ij}^2
-\frac{T^2}{N}
-SS_A-SS_B
$$

Using the four cell totals:

$$
28,\;29,\;33,\;47
$$

we obtain:

$$
SS_{AB}=14.08
$$

---

## 5. Total Sum of Squares

$$
SS_T=\sum Y^2-\frac{T^2}{N}
$$

Therefore:

$$
SS_T=234.92
$$

---

## 6. Error Sum of Squares

$$
SS_E=SS_T-SS_A-SS_B-SS_{AB}
$$

$$
SS_E=158.00
$$

---

# Degrees of Freedom

For StudyGroup:

$$
df_A=a-1=1
$$

For Sex:

$$
df_B=b-1=1
$$

For the interaction:

$$
df_{AB}=(a-1)(b-1)=1
$$

For error:

$$
df_E=ab(n-1)=8
$$

Total:

$$
df_T=N-1=11
$$

---

# Mean Squares

Mean square is calculated as:

$$
MS=\frac{SS}{df}
$$

Therefore:

$$
MS_A=44.08
$$

$$
MS_B=18.75
$$

$$
MS_{AB}=14.08
$$

$$
MS_E=\frac{158}{8}=19.75
$$

---

# F Statistics

The F statistic is:

$$
F=\frac{MS_{\text{effect}}}{MS_E}
$$

### Stu
