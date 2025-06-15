
import pandas as pd

def extract_features(df, window_size=100, step_size=50):
    """
    Extract sliding window statistical features from a cleaned dataframe.

    Args:
        df: pandas DataFrame, must include acceleration and velocity columns.
        window_size: Number of rows in each sliding window.
        step_size: Step size between windows.

    Returns:
        DataFrame with extracted features per window.
    """
    feature_rows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start + window_size]
        features = {}

        # 时间特征
        features['start_time'] = window['Time (s)'].iloc[0]

        # 加速度统计特征
        for col in ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)', 'accel_magnitude']:
            features[f'{col}_mean'] = window[col].mean()
            features[f'{col}_std'] = window[col].std()
            features[f'{col}_min'] = window[col].min()
            features[f'{col}_max'] = window[col].max()

        # 速度统计特征
        for col in ['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)', 'V_magnitude']:
            features[f'{col}_mean'] = window[col].mean()
            features[f'{col}_std'] = window[col].std()

        # 陀螺仪模值
        if 'gyro_magnitude' in window.columns:
            features['gyro_magnitude_mean'] = window['gyro_magnitude'].mean()
            features['gyro_magnitude_std'] = window['gyro_magnitude'].std()

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)
