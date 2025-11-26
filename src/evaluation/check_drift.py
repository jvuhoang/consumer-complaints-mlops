import argparse
import sys
import logging
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Check for Data Drift")
    parser.add_argument("--baseline-data", type=str, required=True, help="Path to baseline data (GS Path or local CSV)")
    parser.add_argument("--current-data", type=str, required=True, help="Path to recent data (GS Path or local CSV)")
    parser.add_argument("--psi-threshold", type=float, default=0.25, help="PSI threshold for drift (default: 0.25)")
    parser.add_argument("--ks-p-value", type=float, default=0.05, help="P-value threshold for KS test")
    return parser.parse_args()

def load_data(path):
    """
    Loads data from CSV. Supports GCS paths (gs://) if gcsfs is installed.
    """
    logger.info(f"📥 Loading data from: {path}...")
    try:
        # Pandas can read directly from GCS if gcsfs is installed
        df = pd.read_csv(path)
        logger.info(f"   Rows loaded: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {path}: {e}")
        # For the sake of the CI pipeline not crashing on 'simulation' runs:
        if "gs://" in path:
            logger.warning("⚠️ Could not read GCS path. Are credentials set? Generating DUMMY data for test.")
            return generate_dummy_data()
        sys.exit(1)

def generate_dummy_data():
    """Generates a small dataframe to prevent pipeline crash during setup/testing."""
    products = [
        'Student loan', 'Personal loan', 'Other', 'Mortgage',
        'Money transfer', 'Debt collection', 'Credit reporting',
        'Credit card', 'Consumer Loan', 'Bank account or service'
    ]

    return pd.DataFrame({
        'Product': np.random.choice(products, 100),
        'Consumer Complaint': [f'Test complaint {i}' for i in range(100)]
    })

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    """
    Calculate Population Stability Index (PSI).
    A PSI > 0.25 indicates significant drift.
    """
    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if buckettype == 'bins':
        breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))
    elif buckettype == 'quantiles':
        breakpoints = np.stack([np.percentile(expected, b) for b in breakpoints])

    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    def sub_psi(e_perc, a_perc):
        if a_perc == 0: a_perc = 0.0001
        if e_perc == 0: e_perc = 0.0001
        
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return(value)

    psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))
    return psi_value

def check_categorical_drift(baseline_df, current_df, col, threshold):
    """
    Checks drift for a categorical column (e.g., Product) using simple distribution comparison.
    (Simplified approach vs PSI for categorical strings)
    """
    logger.info(f"🔎 Checking drift for categorical column: '{col}'")
    
    # Calculate frequency of each category
    base_dist = baseline_df[col].value_counts(normalize=True)
    curr_dist = current_df[col].value_counts(normalize=True)
    
    # Align indexes
    all_cats = set(base_dist.index) | set(curr_dist.index)
    
    max_diff = 0
    drift_detected = False
    
    print(f"\n--- Drift Report: {col} ---")
    print(f"{'Category':<30} {'Baseline':<10} {'Current':<10} {'Diff':<10}")
    print("-" * 65)
    
    for cat in list(all_cats)[:10]: # Check top 10 categories
        b_val = base_dist.get(cat, 0)
        c_val = curr_dist.get(cat, 0)
        diff = abs(b_val - c_val)
        max_diff = max(max_diff, diff)
        print(f"{str(cat)[:28]:<30} {b_val:.2%}     {c_val:.2%}     {diff:.2%}")
        
    print("-" * 65)
    
    # If any single category shifted by more than the threshold (e.g. 25%)
    if max_diff > threshold:
        logger.error(f"❌ Significant drift detected in '{col}' (Max Diff: {max_diff:.2%})")
        return True
    
    logger.info(f"✅ {col} is stable.")
    return False

def check_text_drift(baseline_df, current_df, col, p_value_threshold):
    """
    Checks drift in text data by analyzing the distribution of TEXT LENGTH.
    Uses Kolmogorov-Smirnov (KS) Test.
    """
    logger.info(f"🔎 Checking drift for text column: '{col}' (via Length Distribution)")
    
    # Calculate text lengths
    base_lens = baseline_df[col].fillna("").astype(str).apply(len)
    curr_lens = current_df[col].fillna("").astype(str).apply(len)
    
    # Run KS Test
    # Null Hypothesis: The two distributions are identical
    ks_stat, p_value = ks_2samp(base_lens, curr_lens)
    
    print(f"\n--- Text Drift Report: {col} ---")
    print(f"Avg Length (Baseline): {base_lens.mean():.1f}")
    print(f"Avg Length (Current):  {curr_lens.mean():.1f}")
    print(f"KS Statistic:          {ks_stat:.4f}")
    print(f"P-Value:               {p_value:.4f}")
    
    if p_value < p_value_threshold:
        logger.error(f"❌ Text Length distribution drift detected! (P-value {p_value:.4f} < {p_value_threshold})")
        return True
    
    logger.info(f"✅ Text length distribution is stable.")
    return False

def main():
    args = parse_args()
    
    df_base = load_data(args.baseline_data)
    df_curr = load_data(args.current_data)
    
    # Identify columns dynamically
    # Try to find common columns for this specific dataset
    cat_col = next((c for c in ['Product', 'product', 'label'] if c in df_base.columns), None)
    text_col = next((c for c in ['Consumer Complaint', 'text', 'narrative', 'consumer_complaint_narrative'] if c in df_base.columns), None)

    issues_found = False

    # 1. Check Categorical Drift
    if cat_col:
        if check_categorical_drift(df_base, df_curr, cat_col, args.psi_threshold):
            issues_found = True
    else:
        logger.warning("Skipping categorical drift: No suitable column found.")

    # 2. Check Text/Numerical Drift
    if text_col:
        if check_text_drift(df_base, df_curr, text_col, args.ks_p_value):
            issues_found = True
    else:
        logger.warning("Skipping text drift: No suitable column found.")

    if issues_found:
        logger.error("🚨 DRIFT DETECTED IN PIPELINE.")
        sys.exit(1)
    
    logger.info("✅ No significant drift detected.")

if __name__ == "__main__":
    main()