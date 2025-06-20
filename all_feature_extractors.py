
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

def extract_features(df, window_size=100, step_size=50):
    feature_rows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start + window_size]
        features = {}
        features['start_time'] = window['Time (s)'].iloc[0]

        # Basic Statistics + Advanced Statistics
        for col in ['X_filtered', 'Y_filtered', 'Z_filtered', 'accel_magnitude']:
            if col in window.columns:
                series = window[col].values
                features[f'{col}_mean'] = np.mean(series)
                features[f'{col}_std'] = np.std(series)
                features[f'{col}_min'] = np.min(series)
                features[f'{col}_max'] = np.max(series)
                features[f'{col}_skewness'] = skew(series)
                features[f'{col}_kurtosis'] = kurtosis(series)
                features[f'{col}_rms'] = np.sqrt(np.mean(series**2))
                features[f'{col}_energy'] = np.sum(series**2)
                peaks, _ = find_peaks(series, height=np.std(series))
                features[f'{col}_peak_count'] = len(peaks)

        # V
        for col in ['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)', 'V_magnitude']:
            if col in window.columns:
                features[f'{col}_mean'] = window[col].mean()
                features[f'{col}_std'] = window[col].std()

        # Gyroscope Statistics
        if 'gyro_magnitude' in window.columns:
            features['gyro_magnitude_mean'] = window['gyro_magnitude'].mean()
            features['gyro_magnitude_std'] = window['gyro_magnitude'].std()

        for col in ['X_gyro_filtered', 'Y_gyro_filtered', 'Z_gyro_filtered']:
            if col in window.columns:
                series = window[col].values
                features[f'{col}_mean'] = np.mean(series)
                features[f'{col}_std'] = np.std(series)
                features[f'{col}_min'] = np.min(series)
                features[f'{col}_max'] = np.max(series)
                features[f'{col}_skewness'] = skew(series)
                features[f'{col}_kurtosis'] = kurtosis(series)
                features[f'{col}_rms'] = np.sqrt(np.mean(series**2))
                features[f'{col}_energy'] = np.sum(series**2)
                peaks, _ = find_peaks(series, height=np.std(series))
                features[f'{col}_peak_count'] = len(peaks)

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


def extract_frequency_features(df, window_size=100, step_size=50, sampling_rate=100):
    feature_rows = []

    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start + window_size]
        features = {}
        features['start_time'] = window['Time (s)'].iloc[0]

        for col in ['X_filtered', 'Y_filtered', 'Z_filtered']:
            if col in window.columns:
                signal_data = window[col].values
                fft_result = np.fft.rfft(signal_data)
                fft_magnitude = np.abs(fft_result)
                freqs = np.fft.rfftfreq(len(signal_data), d=1.0 / sampling_rate)

                dominant_idx = np.argmax(fft_magnitude[1:]) + 1
                dominant_freq = freqs[dominant_idx]
                energy = np.sum(fft_magnitude**2) / len(fft_magnitude)
                entropy = -np.sum((fft_magnitude / np.sum(fft_magnitude)) * np.log2(fft_magnitude / np.sum(fft_magnitude) + 1e-12))

                features[f'{col}_dominant_freq'] = dominant_freq
                features[f'{col}_fft_energy'] = energy
                features[f'{col}_fft_entropy'] = entropy

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


def extract_combined_features(df, window_size=100, step_size=50, sampling_rate=100):
    time_features = extract_features(df, window_size, step_size)
    freq_features = extract_frequency_features(df, window_size, step_size, sampling_rate)

    if 'start_time' in freq_features.columns:
        freq_features = freq_features.drop(columns=['start_time'])

    combined = pd.concat([time_features, freq_features], axis=1)
    return combined
