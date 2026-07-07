# ── Imports ───────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# --- Add project root to path so we can import the framework ---
sys.path.insert(0, os.path.dirname(__file__))
from ab_test_framework import (
    run_ab_test,
    detect_outcome_type,
    check_normality
)

# ── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title = "A/B Test Framework",
    page_icon  = "🧪",
    layout     = "wide"
)

# ── Header ────────────────────────────────────────────────────────
st.title("🧪 A/B Test Analysis Framework")
st.markdown(
    "Upload any control/treatment dataset and automatically run "
    "the correct statistical test — t-test, Mann-Whitney U, or "
    "chi-square — with effect sizes and plain-language results."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with control and treatment groups",
        type = ['csv']
    )

    st.divider()

    with st.expander("⚙️ How It Works"):
        st.markdown("""
        **Test Selection Logic:**

        1. **Outcome Type Detection**
           - Boolean/categorical columns → Chi-square test
           - Numeric columns with many unique values → Continuous branch

        2. **Normality Check (Continuous outcomes)**
           - n ≤ 5,000 → Shapiro-Wilk test
           - n > 5,000 → D'Agostino K-squared test
           - Both groups normal → Independent samples t-test
           - Either group non-normal → Mann-Whitney U test

        3. **Effect Sizes**
           - T-test → Cohen's d
           - Mann-Whitney → Rank-biserial correlation
           - Chi-square → Cramer's V

        4. **Confidence Intervals**
           - Continuous → Bootstrap CI (median difference)
           - Categorical → Wilson score CI (proportions)
        """)

    st.divider()
    st.markdown("Built with `scipy.stats`, `pandas`, `streamlit`")

# ── Main Area ─────────────────────────────────────────────────────
if uploaded_file is not None:

    # --- Load data ---
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Data Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", f"{len(df.columns):,}")
    col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    st.dataframe(df.head(10), use_container_width=True)
    st.divider()

    # --- Column selectors ---
    st.subheader("⚙️ Configure Your Test")

    col_left, col_right = st.columns(2)

    with col_left:
        group_col = st.selectbox(
            "Select Group Column (e.g. version, group, treatment)",
            options = df.columns.tolist()
        )

        group_values = df[group_col].unique().tolist()

        group_a = st.selectbox(
            "Select Control Group (Group A)",
            options = group_values
        )

        group_b = st.selectbox(
            "Select Treatment Group (Group B)",
            options = [v for v in group_values if v != group_a]
        )

    with col_right:
        outcome_col = st.selectbox(
            "Select Outcome Column (e.g. retention_7, revenue)",
            options = [c for c in df.columns if c != group_col]
        )

        detected_type = detect_outcome_type(df[outcome_col])
        st.info(
            f"**Detected outcome type:** {detected_type}\n\n"
            f"**Unique values:** {df[outcome_col].nunique()}\n\n"
            f"**dtype:** {df[outcome_col].dtype}"
        )

    st.divider()

    # ── Two Buttons Side by Side ──────────────────────────────────
    btn_left, btn_right = st.columns(2)

    with btn_left:
        run_button = st.button(
            "▶ Run Single Test",
            type = "primary",
            use_container_width = True
        )

    with btn_right:
        run_all_button = st.button(
            "🔁 Run All Outcomes",
            type = "secondary",
            use_container_width = True
        )

    st.divider()

    # ── Helper: Display Single Result ─────────────────────────────
    def display_single_result(result, outcome_name):
        """Displays a single test result with KPIs, verdict, and chart."""

        test_labels = {
            'mann_whitney_u'           : 'Mann-Whitney U Test',
            'independent_samples_ttest': 'Independent Samples T-Test',
            'chi_square'               : 'Chi-Square Test'
        }
        reason_labels = {
            'mann_whitney_u'           : 'continuous outcome, non-normal distribution detected',
            'independent_samples_ttest': 'continuous outcome, normal distribution confirmed',
            'chi_square'               : 'binary or categorical outcome'
        }

        test_name = test_labels.get(result['test_used'], result['test_used'])
        reason    = reason_labels.get(result['test_used'], '')

        st.success(f"**Test selected:** {test_name}  \n**Reason:** {reason}")

        # --- KPI cards ---
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Group A (n)", f"{result['n_a']:,}")
        k2.metric("Group B (n)", f"{result['n_b']:,}")
        k3.metric("p-value", f"{result['p_value']:.4f}")
        k4.metric("Effect Size", f"{result['effect_size']:.4f}")
        k5.metric(
            "Significant?",
            "Yes ✅" if result['significant'] else "No ❌"
        )

        # --- Verdict ---
        if result['significant']:
            st.warning(f"⚠️ {result['verdict']}")
        else:
            st.info(f"ℹ️ {result['verdict']}")

        # --- Detailed results ---
        with st.expander("🔍 Detailed Statistical Results"):
            detail_df = pd.DataFrame([{
                'Outcome'         : outcome_name,
                'Test Used'       : result['test_used'],
                'Statistic'       : result['statistic'],
                'p-value'         : result['p_value'],
                'Effect Size'     : result['effect_size'],
                'Effect Type'     : result['effect_size_type'],
                'Effect Magnitude': result['effect_magnitude'],
                'CI Lower'        : result['confidence_interval'][0],
                'CI Upper'        : result['confidence_interval'][1],
                'Significant'     : result['significant']
            }])
            st.dataframe(detail_df, use_container_width=True)

        # --- Visualization ---
        fig, ax = plt.subplots(figsize=(10, 4))
        group_a_data = df[df[group_col] == group_a][outcome_name]
        group_b_data = df[df[group_col] == group_b][outcome_name]

        if result['outcome_type'] == 'continuous':
            ax.hist(
                group_a_data, bins=50, alpha=0.5,
                color='steelblue', label=f'{group_a}',
                edgecolor='white', density=True
            )
            ax.hist(
                group_b_data, bins=50, alpha=0.5,
                color='seagreen', label=f'{group_b}',
                edgecolor='white', density=True
            )
            ax.axvline(
                group_a_data.median(), color='steelblue',
                linestyle='--', linewidth=2,
                label=f'{group_a} median: {group_a_data.median():.1f}'
            )
            ax.axvline(
                group_b_data.median(), color='seagreen',
                linestyle='--', linewidth=2,
                label=f'{group_b} median: {group_b_data.median():.1f}'
            )
            ax.set_xlabel(outcome_name)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {outcome_name} by Group')
            ax.legend()

        else:
            prop_a = group_a_data.mean() * 100
            prop_b = group_b_data.mean() * 100

            bars = ax.bar(
                [f'{group_a}', f'{group_b}'],
                [prop_a, prop_b],
                color=['steelblue', 'seagreen'],
                edgecolor='white', width=0.4
            )
            for bar, val in zip(bars, [prop_a, prop_b]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f'{val:.2f}%',
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold'
                )
            ax.set_ylabel('Rate (%)')
            ax.set_title(f'{outcome_name} Rate by Group')
            ax.set_ylim(0, max(prop_a, prop_b) * 1.2)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Input Validation Helper ───────────────────────────────────
    def validate_inputs():
        """Returns True if inputs are valid, otherwise shows error."""
        if group_a == group_b:
            st.error("Control and treatment groups must be different.")
            return False
        if len(df[df[group_col] == group_a]) < 10 or \
           len(df[df[group_col] == group_b]) < 10:
            st.error(
                "One or both groups have fewer than 10 observations. "
                "Sample size is too small for reliable testing."
            )
            return False
        return True

    # ── Single Test Execution ─────────────────────────────────────
    if run_button:

        if not validate_inputs():
            st.stop()

        if df[outcome_col].nunique() < 2:
            st.error(
                f"Outcome column '{outcome_col}' has fewer than 2 "
                f"unique values — cannot run a test."
            )
            st.stop()

        with st.spinner("Running statistical test..."):
            try:
                result = run_ab_test(
                    data        = df,
                    group_col   = group_col,
                    outcome_col = outcome_col,
                    group_a     = group_a,
                    group_b     = group_b
                )
            except Exception as e:
                st.error(f"Error running test: {e}")
                st.stop()

        st.subheader(f"📊 Results: {outcome_col}")
        display_single_result(result, outcome_col)

        # --- Download ---
        download_df = pd.DataFrame([{
            'Outcome'         : outcome_col,
            'Test Used'       : result['test_used'],
            'Statistic'       : result['statistic'],
            'p-value'         : result['p_value'],
            'Effect Size'     : result['effect_size'],
            'Effect Type'     : result['effect_size_type'],
            'Effect Magnitude': result['effect_magnitude'],
            'CI Lower'        : result['confidence_interval'][0],
            'CI Upper'        : result['confidence_interval'][1],
            'Significant'     : result['significant'],
            'Verdict'         : result['verdict']
        }])
        st.download_button(
            label     = "⬇️ Download Results as CSV",
            data      = download_df.to_csv(index=False),
            file_name = f"ab_test_{outcome_col}_results.csv",
            mime      = "text/csv"
        )

    # ── Run All Outcomes ──────────────────────────────────────────
    if run_all_button:

        if not validate_inputs():
            st.stop()

        st.subheader("🔁 All Outcomes — Summary")

        valid_outcomes = [
            c for c in df.columns
            if c != group_col and df[c].nunique() >= 2
        ]

        if len(valid_outcomes) == 0:
            st.error("No valid outcome columns found.")
        else:
            all_rows = []
            progress = st.progress(0)
            status   = st.empty()

            for i, oc in enumerate(valid_outcomes):
                status.text(f"Testing {oc}...")
                try:
                    r = run_ab_test(
                        data        = df,
                        group_col   = group_col,
                        outcome_col = oc,
                        group_a     = group_a,
                        group_b     = group_b
                    )
                    all_rows.append({
                        'Outcome'          : oc,
                        'Test Used'        : r['test_used'],
                        'p-value'          : r['p_value'],
                        'Significant?'     : 'Yes ✅' if r['significant'] else 'No ❌',
                        'Effect Size'      : r['effect_size'],
                        'Effect Magnitude' : r['effect_magnitude'],
                        'Verdict'          : r['verdict']
                    })
                except Exception as e:
                    all_rows.append({
                        'Outcome'          : oc,
                        'Test Used'        : 'ERROR',
                        'p-value'          : None,
                        'Significant?'     : 'Error',
                        'Effect Size'      : None,
                        'Effect Magnitude' : None,
                        'Verdict'          : str(e)
                    })

                progress.progress((i + 1) / len(valid_outcomes))

            status.text("✅ All outcomes tested.")
            progress.empty()

            all_results_df = pd.DataFrame(all_rows)
            st.dataframe(all_results_df, use_container_width=True)

            st.download_button(
                label     = "⬇️ Download All Results as CSV",
                data      = all_results_df.to_csv(index=False),
                file_name = "ab_test_all_outcomes_results.csv",
                mime      = "text/csv"
            )

else:
    st.info(
        "👈 Upload a CSV file in the sidebar to get started.\n\n"
        "Don't have a dataset? Try the Cookie Cats dataset from Kaggle: "
        "[cookie_cats.csv](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)"
    )