
import numpy as np
import pandas as pd

def clean_dataframe(df, accel_cols=["acc_x", "acc_y", "acc_z"], z_thresh=3.5):
    """
    Clean data by interpolating missing values and removing outliers based on z-score.

    Args:
        df: pandas DataFrame containing sensor data.
        accel_cols: List of column names to clean (e.g., acceleration columns).
        z_thresh: Z-score threshold for identifying outliers.

    Returns:
        Cleaned pandas DataFrame.
    """
    df_cleaned = df.copy()

    # Step 1: Fill NaN using linear interpolation and ffill/bfill as fallback
    df_cleaned = df_cleaned.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    # Step 2: Outlier detection and removal (based on z-score)
    for col in accel_cols:
        if col in df_cleaned.columns:
            z = (df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std()
            outliers = np.abs(z) > z_thresh
            df_cleaned.loc[outliers, col] = np.nan
            df_cleaned[col] = df_cleaned[col].interpolate().fillna(method='bfill').fillna(method='ffill')

    return df_cleaned
