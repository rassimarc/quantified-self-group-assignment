
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

        for col in ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)']:
            signal_data = window[col].values
            fft_result = np.fft.rfft(signal_data)
            fft_magnitude = np.abs(fft_result)
            freqs = np.fft.rfftfreq(len(signal_data), d=1.0 / sampling_rate)

            # 主频：幅度最大对应的频率
            dominant_idx = np.argmax(fft_magnitude[1:]) + 1  # exclude DC
            dominant_freq = freqs[dominant_idx]
            energy = np.sum(fft_magnitude**2) / len(fft_magnitude)
            entropy = -np.sum((fft_magnitude / np.sum(fft_magnitude)) * np.log2(fft_magnitude / np.sum(fft_magnitude) + 1e-12))

            features[f'{col}_dominant_freq'] = dominant_freq
            features[f'{col}_fft_energy'] = energy
            features[f'{col}_fft_entropy'] = entropy

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)
