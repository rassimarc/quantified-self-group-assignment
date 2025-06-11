import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, fft


def analyze_sample_rate(df, time_column='Time (s)'):
    """
    Analyze the actual sample rate from your data and check for consistency.
    
    Args:
        df: DataFrame with time column
        time_column: Name of the time column
    
    Returns:
        Dictionary with sample rate statistics
    """
    # Calculate time differences between consecutive samples
    time_diffs = df[time_column].diff().dropna()
    
    # Calculate statistics
    mean_dt = time_diffs.mean()
    std_dt = time_diffs.std()
    median_dt = time_diffs.median()
    min_dt = time_diffs.min()
    max_dt = time_diffs.max()
    
    # Calculate sample rates
    mean_sample_rate = 1.0 / mean_dt
    median_sample_rate = 1.0 / median_dt
    
    # Check for regularity (how consistent is the sampling?)
    cv = std_dt / mean_dt  # Coefficient of variation
    
    # Detect outliers (missed samples or delays)
    q1 = time_diffs.quantile(0.25)
    q3 = time_diffs.quantile(0.75)
    iqr = q3 - q1
    outliers = time_diffs[(time_diffs < q1 - 1.5*iqr) | (time_diffs > q3 + 1.5*iqr)]
    
    results = {
        'mean_sample_rate': mean_sample_rate,
        'median_sample_rate': median_sample_rate,
        'mean_dt': mean_dt,
        'std_dt': std_dt,
        'cv': cv,
        'min_dt': min_dt,
        'max_dt': max_dt,
        'outlier_count': len(outliers),
        'total_samples': len(df),
        'time_diffs': time_diffs
    }
    
    # Print summary
    print(f"Sample Rate Analysis:")
    print(f"====================")
    print(f"Mean sample rate: {mean_sample_rate:.2f} Hz")
    print(f"Median sample rate: {median_sample_rate:.2f} Hz")
    print(f"Mean time interval: {mean_dt*1000:.2f} ms")
    print(f"Std dev of intervals: {std_dt*1000:.3f} ms")
    print(f"Coefficient of variation: {cv:.4f} (lower is more regular)")
    print(f"Min/Max intervals: {min_dt*1000:.2f} / {max_dt*1000:.2f} ms")
    print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(time_diffs)*100:.1f}%)")
    
    return results

def plot_sample_rate_analysis(results):
    """
    Visualize the sample rate consistency and distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Time interval distribution
    ax = axes[0, 0]
    time_diffs_ms = results['time_diffs'] * 1000  # Convert to milliseconds
    ax.hist(time_diffs_ms, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(results['mean_dt']*1000, color='red', linestyle='--', 
               label=f'Mean: {results["mean_dt"]*1000:.2f} ms')
    ax.set_xlabel('Time Interval (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Time Intervals Between Samples')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Time intervals over time
    ax = axes[0, 1]
    ax.plot(results['time_diffs'] * 1000, alpha=0.7)
    ax.axhline(results['mean_dt']*1000, color='red', linestyle='--', alpha=0.8)
    ax.fill_between(range(len(results['time_diffs'])), 
                    (results['mean_dt'] - results['std_dt']) * 1000,
                    (results['mean_dt'] + results['std_dt']) * 1000,
                    color='red', alpha=0.2, label='±1 std dev')
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Time Interval (ms)')
    ax.set_title('Time Intervals Throughout Recording')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Cumulative drift from ideal sampling
    ax = axes[1, 0]
    ideal_times = np.arange(len(results['time_diffs'])) * results['mean_dt']
    actual_times = results['time_diffs'].cumsum()
    drift = (actual_times - ideal_times) * 1000  # in milliseconds
    ax.plot(drift)
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Cumulative Drift (ms)')
    ax.set_title('Drift from Ideal Regular Sampling')
    ax.grid(True, alpha=0.3)
    
    # 4. Sample rate variations
    ax = axes[1, 1]
    instantaneous_rates = 1.0 / results['time_diffs']
    ax.plot(instantaneous_rates, alpha=0.7)
    ax.axhline(results['mean_sample_rate'], color='red', linestyle='--', 
               label=f'Mean: {results["mean_sample_rate"]:.1f} Hz')
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Instantaneous Sample Rate (Hz)')
    ax.set_title('Sample Rate Variations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def determine_effective_sample_rate(df, time_column='Time (s)', method='robust_mean'):
    """
    Determine the most appropriate sample rate to use for processing.
    
    Args:
        df: DataFrame with time data
        time_column: Name of time column
        method: 'mean', 'median', 'mode', or 'robust_mean'
    
    Returns:
        Recommended sample rate in Hz
    """
    time_diffs = df[time_column].diff().dropna()
    
    if method == 'mean':
        return 1.0 / time_diffs.mean()
    elif method == 'median':
        return 1.0 / time_diffs.median()
    elif method == 'mode':
        # Find the most common time interval (binned)
        hist, bins = np.histogram(time_diffs, bins=50)
        mode_bin = bins[np.argmax(hist)]
        return 1.0 / mode_bin
    elif method == 'robust_mean':
        # Remove outliers before calculating mean
        q1 = time_diffs.quantile(0.25)
        q3 = time_diffs.quantile(0.75)
        iqr = q3 - q1
        filtered = time_diffs[(time_diffs >= q1 - 1.5*iqr) & (time_diffs <= q3 + 1.5*iqr)]
        return 1.0 / filtered.mean()
    
def check_nyquist_frequency(sample_rate, expected_max_frequency=None):
    """
    Check if the sample rate is appropriate for the expected signal content.
    
    Args:
        sample_rate: Sample rate in Hz
        expected_max_frequency: Maximum frequency expected in the signal
    """
    nyquist = sample_rate / 2
    
    print(f"\nNyquist Frequency Analysis:")
    print(f"===========================")
    print(f"Sample rate: {sample_rate:.2f} Hz")
    print(f"Nyquist frequency: {nyquist:.2f} Hz")
    print(f"Can accurately capture frequencies up to: {nyquist:.2f} Hz")
    
    # Common motion frequencies
    print(f"\nCommon motion frequency ranges:")
    print(f"- Walking: 0.5 - 3 Hz")
    print(f"- Running: 2 - 4 Hz")
    print(f"- Vibrations: 10 - 50 Hz")
    print(f"- High-freq tremor: 4 - 12 Hz")
    
    if sample_rate >= 100:
        print(f"\n✓ {sample_rate:.0f} Hz is good for most human motion capture")
    elif sample_rate >= 50:
        print(f"\n⚠ {sample_rate:.0f} Hz is adequate for basic motion, may miss fast movements")
    else:
        print(f"\n✗ {sample_rate:.0f} Hz may be too low for accurate motion capture")
    
    if expected_max_frequency:
        if sample_rate >= 2 * expected_max_frequency:
            print(f"\n✓ Sample rate is adequate for max frequency of {expected_max_frequency} Hz")
        else:
            recommended = 2.5 * expected_max_frequency  # 2.5x for safety margin
            print(f"\n✗ Sample rate too low! Need at least {2*expected_max_frequency:.1f} Hz")
            print(f"   Recommended: {recommended:.1f} Hz")

def analyze_frequency_content(df, accel_columns, sample_rate):
    """
    Analyze the frequency content of your acceleration data using FFT.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for idx, col in enumerate(accel_columns[:3]):
        ax = axes[idx//2, idx%2]
        
        # Perform FFT
        signal = df[col].values
        n = len(signal)
        fft_vals = fft.fft(signal)
        fft_freq = fft.fftfreq(n, 1/sample_rate)
        
        # Only plot positive frequencies
        pos_mask = fft_freq > 0
        frequencies = fft_freq[pos_mask]
        magnitudes = np.abs(fft_vals[pos_mask])
        
        # Plot
        ax.semilogy(frequencies, magnitudes, alpha=0.7)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'Frequency Content - {col}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, sample_rate/2)  # Up to Nyquist
        
        # Mark important frequencies
        ax.axvline(x=sample_rate/2, color='red', linestyle='--', 
                  alpha=0.5, label='Nyquist frequency')
        ax.legend()
    
    # Combined spectrum
    ax = axes[1, 1]
    for col in accel_columns:
        signal = df[col].values
        fft_vals = fft.fft(signal)
        magnitudes = np.abs(fft_vals[pos_mask])
        ax.semilogy(frequencies, magnitudes, alpha=0.5, label=col)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('Combined Frequency Content')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(50, sample_rate/2))  # Focus on lower frequencies
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Using your data structure
    print("=== SAMPLE RATE ANALYSIS FOR YOUR DATA ===\n")
    
    acceleration_file = "acceleration_with_times.csv"

    # Load your data
    sample_data = pd.read_csv(acceleration_file)
    
    df = pd.DataFrame(sample_data)
    
    # Analyze sample rate
    results = analyze_sample_rate(df)
    
    # Plot analysis
    plot_sample_rate_analysis(results)
    
    # Determine best sample rate to use
    recommended_rate = determine_effective_sample_rate(df, method='robust_mean')
    print(f"\nRecommended sample rate for processing: {recommended_rate:.1f} Hz")
    
    # Check if it's appropriate
    check_nyquist_frequency(recommended_rate)
    
    print("\n=== WHAT THIS MEANS FOR YOUR DATA ===")
    print(f"Your data appears to be sampled at approximately {recommended_rate:.0f} Hz")
    print(f"Time interval between samples: ~{1000/recommended_rate:.1f} ms")
    
    if results['cv'] < 0.01:
        print("✓ Very regular sampling - excellent for integration")
    elif results['cv'] < 0.05:
        print("✓ Reasonably regular sampling - good for integration")
    else:
        print("⚠ Irregular sampling detected - may need resampling")