# 🧪 A/B Test Analysis Framework

A end-to-end data science project that builds a reusable statistical 
testing framework in Python, validated on the Cookie Cats mobile game 
A/B test dataset (90,189 players).

## 📌 Project Overview

This project analyzes a randomized experiment testing whether moving 
a level gate from level 30 to level 40 in the mobile game Cookie Cats 
affects player engagement and retention.

Beyond the analysis itself, the project produces a reusable Python 
framework that automatically selects and runs the correct statistical 
test on any two-group A/B test dataset — t-test, Mann-Whitney U, or 
chi-square — with effect sizes, confidence intervals, and plain-language 
verdicts.

The framework is demonstrated through an interactive Streamlit app that 
accepts any uploaded CSV and returns results instantly.


---

## 📊 Dataset

**Cookie Cats A/B Test** — publicly available on 
[Kaggle](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)

| Column | Description |
|---|---|
| userid | Unique player identifier |
| version | Group assignment (gate_30 = control, gate_40 = treatment) |
| sum_gamerounds | Total game rounds played |
| retention_1 | Returned 1 day after installing (bool) |
| retention_7 | Returned 7 days after installing (bool) |

---

## 🗂️ Project Structure

```
ab-test-framework/
├── ab_test_framework.py        ← Reusable statistical testing framework
├── app.py                      ← Interactive Streamlit app
├── cookie_cats.csv             ← Raw dataset
├── requirements.txt            ← Python dependencies
├── data/
│   ├── cookie_cats_cleaned.csv ← Cleaned dataset (outlier removed)
│   ├── phase4_results_summary.csv
│   ├── phase4_interpretation.txt
│   └── phase4_recommendation.txt
├── notebooks/
│   ├── phase1_data_cleaning.ipynb
│   ├── phase2_eda_test_selection.ipynb
│   ├── phase3_framework_validation.ipynb
│   └── phase4_interpretation_recommendation.ipynb
└── sql/
    └── phase1_data_quality_checks.sql
```

---

## 🔬 Project Phases

### Phase 1 — Data Cleaning & SQL Audit
- Loaded raw CSV into SQLite database
- SQL-based quality checks: duplicates, nulls, group balance
- Identified and removed extreme outlier (userid 6390605, 
  49,854 rounds — 101x above 99th percentile)
- Produced cleaned dataset of 90,188 rows

### Phase 2 — Exploratory Analysis & Test Selection
- Visualized distributions: histograms, boxplots, Q-Q plots
- Formally tested normality via Shapiro-Wilk (W≈0.49, p≈10⁻⁸⁰)
- Selected correct tests with justification:
  - **Mann-Whitney U** for sum_gamerounds (non-normal distribution)
  - **Chi-square** for retention_1 and retention_7 (binary outcomes)
- Calculated effect sizes alongside every p-value

### Phase 3 — Reusable Framework (`ab_test_framework.py`)
- Single public function: `run_ab_test(data, group_col, outcome_col, group_a, group_b)`
- Automatically detects outcome type (continuous vs categorical)
- Selects correct test based on normality check
- Calculates effect sizes (Cohen's d, rank-biserial, Cramer's V)
- Returns bootstrap CI, Wilson score CI, and plain-language verdict
- Validated against Phase 2 manual results exactly
- Tested on synthetic normal data to confirm t-test branch works

### Phase 4 — Business Interpretation
- Consolidated all three results into one summary table
- Explicitly separated statistical significance from practical importance
- Wrote clear one-sentence business recommendation
- Documented limitations honestly (novelty effects, no segmentation,
  no monetization data, snapshot only)

### Phase 5 — Streamlit App (`app.py`)
- Upload any control/treatment CSV
- Dynamic column selectors — not hardcoded to Cookie Cats
- Auto-selects correct statistical test and displays results
- Distribution visualizations update based on uploaded data
- Single outcome and all-outcomes-at-once execution
- Downloadable CSV results


---

## 📈 Key Findings

| Outcome | Test | p-value | Significant? | Effect Size | Practical Impact |
|---|---|---|---|---|---|
| sum_gamerounds | Mann-Whitney U | 0.0509 | No ❌ | -0.0075 (negligible) | None |
| retention_1 | Chi-Square | 0.0750 | No ❌ | 0.0059 (negligible) | None |
| retention_7 | Chi-Square | 0.0016 | **Yes ✅** | 0.0105 (negligible) | Marginal |

### What This Means:
- **Engagement (sum_gamerounds):** No significant difference between 
  groups — gate position does not affect how many rounds players play
- **1-day retention:** No significant difference — gate position does 
  not affect whether players return the next day
- **7-day retention:** Statistically significant difference in favor 
  of gate_30 — but effect size is negligible (Cramer's V = 0.0105)
- **gate_30 outperforms gate_40 on all three metrics**

---

## 💡 Business Recommendation

> *"Based on a properly randomized experiment of 90,188 players,
> moving the gate from level 30 to level 40 shows no improvement
> in player engagement or 1-day retention, and produces a
> statistically significant reduction in 7-day retention
> (19.02% vs 18.20%, p=0.0016). While the effect is modest,
> gate_30 outperforms gate_40 on every measured outcome.
> Moving the gate is not recommended."*

---

## ⚠️ Limitations

1. **Novelty effect** — behavior change may reflect novelty, 
   not gate position
2. **No segmentation** — new vs returning players not analyzed 
   separately
3. **Snapshot only** — retention measured at day 1 and day 7 only
4. **No monetization data** — revenue impact unknown
5. **Small effect sizes** — all effects negligible even where 
   significant
6. **External validity** — results specific to this game and 
   time period

---

## 🛠️ How to Run

1. Clone the repository using `git clone https://github.com/kunalchouhan2004-ship-it/ab-test-framework.git` and move into the project folder with `cd ab-test-framework`.

2. Install all required dependencies using `pip install -r requirements.txt`.

3. Launch the interactive Streamlit dashboard by running `streamlit run app.py`, which will open the app in your browser and let you upload any control/treatment CSV for analysis.

4. To use the framework directly in Python, import `run_ab_test` from `ab_test_framework.py`, load your dataset with pandas, and pass the dataframe, group column, outcome column, and group labels into the function to automatically select the correct statistical test and return the p-value, effect size, confidence interval, and plain-language verdict.


---

## 🔧 Framework — Test Selection Logic

```
Outcome column
│
├── bool / categorical dtype → Chi-Square Test
│                               Effect size: Cramer's V
│
└── numeric (continuous)
    │
    ├── Normality check
    │   ├── n ≤ 5,000 → Shapiro-Wilk
    │   └── n > 5,000 → D'Agostino K-squared
    │
    ├── Both groups normal → Independent Samples T-Test
    │                         Effect size: Cohen's d
    │
    └── Either non-normal → Mann-Whitney U Test
                             Effect size: Rank-biserial r
```


---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas | Data manipulation |
| scipy.stats | Statistical tests |
| numpy | Numerical computation |
| matplotlib / seaborn | Visualization |
| sqlite3 | SQL data quality checks |
| streamlit | Interactive app |

---

## 👤 Author

Built as a portfolio project demonstrating end-to-end data science:
SQL auditing → statistical analysis → reusable framework →
business interpretation → interactive app.