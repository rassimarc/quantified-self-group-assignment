
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

        # Time feature
        features['start_time'] = window['Time (s)'].iloc[0]

        # Acceleration statistical feature
        for col in ['X_filtered', 'Y_filtered', 'Z_filtered', 'accel_magnitude']:
            features[f'{col}_mean'] = window[col].mean()
            features[f'{col}_std'] = window[col].std()
            features[f'{col}_min'] = window[col].min()
            features[f'{col}_max'] = window[col].max()

        # Speed ​​Statistical Features
        for col in ['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)', 'V_magnitude']:
            features[f'{col}_mean'] = window[col].mean()
            features[f'{col}_std'] = window[col].std()

        # Gyroscope module
        if 'gyro_magnitude' in window.columns:
            features['gyro_magnitude_mean'] = window['gyro_magnitude'].mean()
            features['gyro_magnitude_std'] = window['gyro_magnitude'].std()

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)



import numpy as np
import pandas as pd

def extract_frequency_features(df, window_size=100, step_size=50, sampling_rate=100):
    """
    Extract frequency-domain features using FFT from acceleration signals.

    Args:
        df: DataFrame containing acceleration columns.
        window_size: Number of rows in each window.
        step_size: Sliding step between windows.
        sampling_rate: Hz, used to convert frequency bin indices.

    Returns:
        DataFrame of frequency features per window.
    """
    feature_rows = []

    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start + window_size]
        features = {}
        features['start_time'] = window['Time (s)'].iloc[0]

        for col in ['X_filtered', 'Y_filtered', 'Z_filtered']:
            signal_data = window[col].values
            fft_result = np.fft.rfft(signal_data)
            fft_magnitude = np.abs(fft_result)
            freqs = np.fft.rfftfreq(len(signal_data), d=1.0 / sampling_rate)

            # Main frequency: the frequency corresponding to the maximum amplitude
            dominant_idx = np.argmax(fft_magnitude[1:]) + 1  # exclude DC
            dominant_freq = freqs[dominant_idx]
            energy = np.sum(fft_magnitude**2) / len(fft_magnitude)
            entropy = -np.sum((fft_magnitude / np.sum(fft_magnitude)) * np.log2(fft_magnitude / np.sum(fft_magnitude) + 1e-12))

            features[f'{col}_dominant_freq'] = dominant_freq
            features[f'{col}_fft_energy'] = energy
            features[f'{col}_fft_entropy'] = entropy

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)



import pandas as pd
from extract_features import extract_features
from extract_frequency_features import extract_frequency_features

def extract_combined_features(df, window_size=100, step_size=50, sampling_rate=100):
    """
    Extract combined time-domain and frequency-domain features from dataframe.

    Args:
        df: Cleaned input DataFrame with acceleration and velocity columns.
        window_size: Sliding window size (in samples).
        step_size: Step size between windows.
        sampling_rate: Sampling rate in Hz for frequency features.

    Returns:
        DataFrame containing combined features.
    """
    time_features = extract_features(df, window_size, step_size)
    freq_features = extract_frequency_features(df, window_size, step_size, sampling_rate)

    # Drop duplicate 'start_time' from freq_features to avoid column conflict
    if 'start_time' in freq_features.columns:
        freq_features = freq_features.drop(columns=['start_time'])

    combined = pd.concat([time_features, freq_features], axis=1)
    return combined

# if __name__ == "__main__":
    df = pd.read_csv("xiyan_walk1.csv")
    features = extract_combined_features(df)
    features.to_csv("combined_features_walk1.csv", index=False)
    print("Combined feature file saved as combined_features_walk1.csv")
