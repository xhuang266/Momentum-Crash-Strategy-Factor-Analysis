import matplotlib.pyplot as plt
import warnings
from src.part0_data import download_and_prepare_data
from src.part1_hypothesis import run_hypothesis_validation
from src.part2_baseline import create_baseline_strategy
from src.part4_original import run_sensitivity_analysis
from src.part5_xgboost import run_xgboost_strategy
from src.part6_comparison import run_final_comparison
from src.part7_arima import run_volume_forecasting

# --- Global Settings ---
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')

def main():
    print("===========================================================================")
    print("   Momentum Crash Strategy & Factor Analysis Pipeline  ")
    print("===========================================================================")

    # Part 0: Data Download & Feature Engineering
    df = download_and_prepare_data(start_date='2013-04-18', end_date='2025-6-24')

    # Part 1: Core Hypothesis Validation
    run_hypothesis_validation(df)

    # Part 2: Baseline Strategy (Market Neutral)
    df = create_baseline_strategy(df)

    # Part 4: Original Strategy Sensitivity Analysis
    # Note: Returns robust parameters for Part 6
    robust_period, robust_multiplier = run_sensitivity_analysis(df, commission_bps=1, slippage_bps=1)

    # Part 5: Upgraded Strategy (XGBoost Classifier)
    df = run_xgboost_strategy(df)

    # Part 6: Final Strategy Comparison
    run_final_comparison(df, robust_period, robust_multiplier, commission_bps=1, slippage_bps=1)

    # Part 7: Volume Forecasting (ARIMA)
    run_volume_forecasting(df)

    print("\n===========================================================================")
    print("                       PIPELINE EXECUTION COMPLETED                        ")
    print("===========================================================================")

if __name__ == "__main__":
    main()