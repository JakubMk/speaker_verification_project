import matplotlib.pyplot as plt
import pandas as pd

def plot_eer_curve(eer_df: pd.DataFrame) -> None:
    """
    Plots the Equal Error Rate (EER) curve given a DataFrame with FAR, FRR, and threshold columns.

    Args:
        eer_df (pd.DataFrame): DataFrame containing at least the columns 'margin', 'FAR', 'FRR', and 'diff',
                               as output from the EER evaluation process.

    Returns:
        None. Displays the plot.
    """
    # Find the row with minimal difference between FAR and FRR (the EER threshold)
    eer_row = eer_df.loc[eer_df['diff'].idxmin()]

    plt.figure(figsize=(7, 5))
    plt.plot(eer_df["margin"], eer_df["FAR"], label="FAR", lw=2)
    plt.plot(eer_df["margin"], eer_df["FRR"], label="FRR", lw=2)
    plt.axvline(eer_row["margin"], color='gray', linestyle='--', label=f"EER: {eer_row['FAR']:.3f}")

    plt.xlabel("Threshold (margin)")
    plt.ylabel("Error Rate")
    plt.title("Equal Error Rate (EER) Curve")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()
