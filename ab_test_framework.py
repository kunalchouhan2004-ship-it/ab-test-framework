"""
ab_test_framework.py
====================
A reusable two-group A/B test framework that automatically:
  - Detects whether the outcome variable is continuous or categorical
  - Selects the appropriate statistical test based on normality
  - Calculates effect sizes alongside p-values
  - Returns structured results with a plain-language verdict

Supported tests:
  - Independent samples t-test     (continuous, normal)
  - Mann-Whitney U test            (continuous, non-normal)
  - Chi-square test                (binary/categorical)

Usage:
  from ab_test_framework import run_ab_test
  result = run_ab_test(
      data        = df,
      group_col   = 'version',
      outcome_col = 'sum_gamerounds',
      group_a     = 'gate_30',
      group_b     = 'gate_40'
  )
  print(result['verdict'])

Returns:
  A dictionary containing:
    - outcome_type        : 'continuous' or 'categorical'
    - test_used           : name of the test run
    - statistic           : test statistic value
    - p_value             : p-value of the test
    - effect_size         : effect size value
    - effect_size_type    : name of the effect size metric used
    - effect_magnitude    : 'negligible', 'small', 'medium', or 'large'
    - confidence_interval : (lower, upper) tuple
    - significant         : True or False
    - verdict             : plain-language summary string
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import (
    shapiro, normaltest, ttest_ind,
    mannwhitneyu, chi2_contingency
)

# ── Configuration ──────────────────────────────────────────────────
ALPHA_DEFAULT         = 0.05   # default significance threshold
NORMALITY_SAMPLE_SIZE = 5000   # max sample size for Shapiro-Wilk
                                # above this we switch to D'Agostino
CONTINUOUS_UNIQUE_MIN = 10     # minimum unique values to treat
                                # outcome as continuous vs categorical

print("ab_test_framework.py loaded successfully.")


# ── Helper: Outcome Type Detection ────────────────────────────────

def detect_outcome_type(series):
    """
    Detects whether a pandas Series is continuous or categorical.

    Rules:
      - bool dtype                                       → categorical
      - object/string dtype                              → categorical
      - numeric with <= CONTINUOUS_UNIQUE_MIN unique values → categorical
      - numeric with >  CONTINUOUS_UNIQUE_MIN unique values → continuous

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    str : 'continuous' or 'categorical'
    """
    if series.dtype == bool:
        return 'categorical'
    if series.dtype == object:
        return 'categorical'
    if series.nunique() <= CONTINUOUS_UNIQUE_MIN:
        return 'categorical'
    return 'continuous'


# ── Helper: Normality Check ───────────────────────────────────────

def check_normality(series, alpha=ALPHA_DEFAULT):
    """
    Tests whether a numeric Series is normally distributed.

    Uses Shapiro-Wilk for n <= NORMALITY_SAMPLE_SIZE,
    D'Agostino K-squared for larger samples.

    Parameters
    ----------
    series : pd.Series
    alpha  : float

    Returns
    -------
    dict : is_normal, test_used, statistic, p_value
    """
    n = len(series)

    if n <= NORMALITY_SAMPLE_SIZE:
        stat, p = shapiro(series)
        test_used = 'shapiro'
    else:
        sample = series.sample(NORMALITY_SAMPLE_SIZE, random_state=42)
        stat, p = normaltest(sample)
        test_used = 'dagostino'

    return {
        'is_normal' : p > alpha,
        'test_used' : test_used,
        'statistic' : round(float(stat), 6),
        'p_value'   : round(float(p), 6)
    }


# ── Helper: Effect Size Magnitude Label ───────────────────────────

def effect_magnitude(effect_size, metric='cohens_d'):
    """
    Returns a plain-language label for effect size magnitude.

    Benchmarks:
      Cohen's d       : 0.2=small, 0.5=medium, 0.8=large
      Rank-biserial r : 0.1=small, 0.3=medium, 0.5=large
      Cramer's V      : 0.1=small, 0.3=medium, 0.5=large
    """
    abs_e = abs(effect_size)

    if metric == 'cohens_d':
        if abs_e < 0.2:   return 'negligible'
        elif abs_e < 0.5: return 'small'
        elif abs_e < 0.8: return 'medium'
        else:             return 'large'

    elif metric in ('rank_biserial', 'cramers_v'):
        if abs_e < 0.1:   return 'negligible'
        elif abs_e < 0.3: return 'small'
        elif abs_e < 0.5: return 'medium'
        else:             return 'large'

    return 'unknown'


# ── Helper: Bootstrap Confidence Interval ─────────────────────────

def bootstrap_ci(group_a, group_b, n_boot=1000, ci=95, random_state=42):
    """
    Bootstrap CI for the difference in medians (group_a - group_b).

    Parameters
    ----------
    group_a, group_b : array-like
    n_boot           : int
    ci               : float
    random_state     : int

    Returns
    -------
    tuple : (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_state)
    a = np.array(group_a)
    b = np.array(group_b)
    diffs = []

    for _ in range(n_boot):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.median(sample_a) - np.median(sample_b))

    lower = np.percentile(diffs, (100 - ci) / 2)
    upper = np.percentile(diffs, 100 - (100 - ci) / 2)
    return (round(float(lower), 4), round(float(upper), 4))


# ── Core: Continuous Outcome Branch ───────────────────────────────

def run_continuous_test(group_a, group_b, alpha=ALPHA_DEFAULT):
    """
    Runs t-test or Mann-Whitney U depending on normality.

    Parameters
    ----------
    group_a, group_b : pd.Series
    alpha            : float

    Returns
    -------
    dict : structured results
    """
    norm_a = check_normality(group_a, alpha)
    norm_b = check_normality(group_b, alpha)
    both_normal = norm_a['is_normal'] and norm_b['is_normal']

    if both_normal:
        stat, p = ttest_ind(group_a, group_b)
        test_used = 'independent_samples_ttest'
        pooled_std = np.sqrt(
            (group_a.std() ** 2 + group_b.std() ** 2) / 2
        )
        es = (group_a.mean() - group_b.mean()) / pooled_std
        es_type = 'cohens_d'
    else:
        stat, p = mannwhitneyu(group_a, group_b, alternative='two-sided')
        test_used = 'mann_whitney_u'
        n1, n2 = len(group_a), len(group_b)
        es = 1 - (2 * stat) / (n1 * n2)
        es_type = 'rank_biserial'

    ci = bootstrap_ci(group_a, group_b)

    return {
        'outcome_type'       : 'continuous',
        'test_used'          : test_used,
        'normality_a'        : norm_a,
        'normality_b'        : norm_b,
        'statistic'          : round(float(stat), 4),
        'p_value'            : round(float(p), 6),
        'effect_size'        : round(float(es), 6),
        'effect_size_type'   : es_type,
        'effect_magnitude'   : effect_magnitude(es, es_type),
        'confidence_interval': ci,
        'significant'        : p < alpha
    }


# ── Helper: Wilson Score Confidence Interval ──────────────────────

def wilson_ci(successes, n, ci=95):
    """
    Wilson score CI for a proportion.

    Parameters
    ----------
    successes : int
    n         : int
    ci        : float

    Returns
    -------
    tuple : (lower_bound, upper_bound)
    """
    alpha = 1 - ci / 100
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    center = (p_hat + z**2 / (2*n)) / (1 + z**2 / n)
    margin = (z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / (1 + z**2/n)
    return (round(float(center - margin), 6), round(float(center + margin), 6))


# ── Core: Categorical Outcome Branch ──────────────────────────────

def run_categorical_test(data, group_col, outcome_col,
                         group_a, group_b, alpha=ALPHA_DEFAULT):
    """
    Runs chi-square test for binary/categorical outcomes.

    Parameters
    ----------
    data        : pd.DataFrame
    group_col   : str
    outcome_col : str
    group_a     : str
    group_b     : str
    alpha       : float

    Returns
    -------
    dict : structured results
    """
    ct = pd.crosstab(data[group_col], data[outcome_col])
    chi2, p, dof, expected = chi2_contingency(ct)

    n = len(data)
    k = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k))

    group_a_data = data[data[group_col] == group_a][outcome_col]
    group_b_data = data[data[group_col] == group_b][outcome_col]

    prop_a   = group_a_data.mean()
    prop_b   = group_b_data.mean()
    prop_diff = prop_a - prop_b

    n_a    = len(group_a_data)
    n_b    = len(group_b_data)
    succ_a = int(group_a_data.sum())
    succ_b = int(group_b_data.sum())

    ci_a = wilson_ci(succ_a, n_a)
    ci_b = wilson_ci(succ_b, n_b)

    z = stats.norm.ppf(0.975)
    se_diff = np.sqrt(
        prop_a * (1 - prop_a) / n_a +
        prop_b * (1 - prop_b) / n_b
    )
    ci_diff = (
        round(float(prop_diff - z * se_diff), 6),
        round(float(prop_diff + z * se_diff), 6)
    )

    return {
        'outcome_type'       : 'categorical',
        'test_used'          : 'chi_square',
        'contingency_table'  : ct,
        'statistic'          : round(float(chi2), 4),
        'p_value'            : round(float(p), 6),
        'degrees_of_freedom' : dof,
        'effect_size'        : round(float(cramers_v), 6),
        'effect_size_type'   : 'cramers_v',
        'effect_magnitude'   : effect_magnitude(cramers_v, 'cramers_v'),
        'prop_a'             : round(float(prop_a), 6),
        'prop_b'             : round(float(prop_b), 6),
        'prop_diff'          : round(float(prop_diff), 6),
        'ci_group_a'         : ci_a,
        'ci_group_b'         : ci_b,
        'confidence_interval': ci_diff,
        'significant'        : p < alpha
    }


# ── Helper: Plain Language Verdict ────────────────────────────────

def generate_verdict(result, group_a, group_b, outcome_col, alpha):
    """
    Converts numeric results into a plain-language summary.

    Parameters
    ----------
    result      : dict
    group_a     : str
    group_b     : str
    outcome_col : str
    alpha       : float

    Returns
    -------
    str
    """
    test_map = {
        'mann_whitney_u'           : 'Mann-Whitney U test',
        'independent_samples_ttest': 'independent samples t-test',
        'chi_square'               : 'Chi-square test'
    }
    reason_map = {
        'mann_whitney_u'           : 'non-normal distribution detected',
        'independent_samples_ttest': 'normal distribution confirmed',
        'chi_square'               : 'binary/categorical outcome'
    }

    test_name   = test_map.get(result['test_used'], result['test_used'])
    reason      = reason_map.get(result['test_used'], '')
    p_value     = result['p_value']
    es          = result['effect_size']
    es_type     = result['effect_size_type']
    magnitude   = result['effect_magnitude']
    significant = result['significant']
    ci          = result['confidence_interval']

    if significant:
        sig_statement = (
            f"the difference between {group_a} and {group_b} was "
            f"STATISTICALLY SIGNIFICANT (p={p_value:.4f}, alpha={alpha})"
        )
    else:
        sig_statement = (
            f"the difference between {group_a} and {group_b} was "
            f"NOT statistically significant (p={p_value:.4f}, alpha={alpha})"
        )

    es_statement = (
        f"Effect size was {magnitude} ({es_type}={es:.4f})"
    )

    extra = ''
    if result['outcome_type'] == 'categorical':
        extra = (
            f" {group_a} rate: {result['prop_a']*100:.2f}%, "
            f"{group_b} rate: {result['prop_b']*100:.2f}% "
            f"(difference: {result['prop_diff']*100:.2f} pp)."
        )
    elif result['outcome_type'] == 'continuous':
        extra = (
            f" 95% bootstrap CI for median difference: "
            f"[{float(ci[0]):.2f}, {float(ci[1]):.2f}]."
        )

    return (
        f"[{outcome_col}] Using a {test_name} ({reason}), "
        f"{sig_statement}. "
        f"{es_statement}.{extra}"
    )


# ── Main Public Function ───────────────────────────────────────────

def run_ab_test(data, group_col, outcome_col,
                group_a, group_b, alpha=ALPHA_DEFAULT):
    """
    Main entry point for the A/B test framework.

    Parameters
    ----------
    data        : pd.DataFrame
    group_col   : str
    outcome_col : str
    group_a     : str
    group_b     : str
    alpha       : float

    Returns
    -------
    dict : full structured results including verdict
    """
    if group_col not in data.columns:
        raise ValueError(f"group_col '{group_col}' not found in data.")
    if outcome_col not in data.columns:
        raise ValueError(f"outcome_col '{outcome_col}' not found in data.")
    if group_a not in data[group_col].values:
        raise ValueError(f"group_a '{group_a}' not found in {group_col}.")
    if group_b not in data[group_col].values:
        raise ValueError(f"group_b '{group_b}' not found in {group_col}.")

    group_a_data = data[data[group_col] == group_a][outcome_col]
    group_b_data = data[data[group_col] == group_b][outcome_col]
    outcome_type = detect_outcome_type(data[outcome_col])

    if outcome_type == 'continuous':
        result = run_continuous_test(group_a_data, group_b_data, alpha)
    else:
        result = run_categorical_test(
            data, group_col, outcome_col, group_a, group_b, alpha
        )

    verdict = generate_verdict(result, group_a, group_b, outcome_col, alpha)

    result['verdict']     = verdict
    result['alpha']       = alpha
    result['group_a']     = group_a
    result['group_b']     = group_b
    result['outcome_col'] = outcome_col
    result['n_a']         = len(group_a_data)
    result['n_b']         = len(group_b_data)

    return result