import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
from scipy.integrate import cumulative_trapezoid as cumtrapz

def analyze_frequency_for_cutoff(df, accel_columns=['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'], 
                                sample_rate=100, plot=True):
    """
    Analyze frequency content to help choose highpass cutoff.
    
    Returns:
        Dictionary with suggested cutoff frequencies
    """
    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10)) if plot else None
    
    # 1. Power Spectral Density Analysis
    ax = axes[0, 0] if plot else None
    
    for col in accel_columns:
        # Calculate PSD
        freqs, psd = signal.welch(df[col].values, fs=sample_rate, nperseg=min(256, len(df)//4))
        
        if plot:
            ax.semilogy(freqs, psd, label=col, alpha=0.7)
        
        # Find frequency where 99% of power is above
        cumulative_power = np.cumsum(psd)
        total_power = cumulative_power[-1]
        idx_99 = np.where(cumulative_power >= 0.01 * total_power)[0][0]
        results[f'{col}_99_power_freq'] = freqs[idx_99]
    
    if plot:
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title('PSD of Acceleration Signals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)  # Focus on low frequencies
    
    # 2. Integrated Signal Analysis (Velocity Drift)
    ax = axes[0, 1] if plot else None
    
    # Test different cutoff frequencies
    cutoff_frequencies = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(cutoff_frequencies)))
    
    drift_values = []
    
    for i, cutoff in enumerate(cutoff_frequencies):
        if cutoff < sample_rate / 2:  # Valid cutoff
            # Design filter
            b, a = signal.butter(2, cutoff / (sample_rate / 2), btype='high')
            
            # Apply to one axis as example
            filtered = signal.filtfilt(b, a, df[accel_columns[0]].values)
            
            # Integrate to get velocity
            dt = 1.0 / sample_rate
            velocity = cumtrapz(filtered, dx=dt, initial=0)
            
            # Calculate drift (final velocity)
            drift = abs(velocity[-1])
            drift_values.append((cutoff, drift))
            
            if plot:
                time = np.arange(len(velocity)) * dt
                ax.plot(time, velocity, color=colors[i], 
                       label=f'{cutoff} Hz', alpha=0.7)
    
    if plot:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity Drift with Different Highpass Cutoffs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Motion Detection Analysis
    ax = axes[1, 0] if plot else None
    
    # Calculate acceleration magnitude
    accel_mag = np.sqrt(sum(df[col]**2 for col in accel_columns))
    
    # Subtract gravity (approximately)
    accel_mag_no_g = accel_mag - accel_mag.mean()
    
    # Find periods of motion vs. stationary
    window_size = int(0.5 * sample_rate)  # 0.5 second windows
    rolling_std = pd.Series(accel_mag_no_g).rolling(window_size).std()
    motion_threshold = rolling_std.quantile(0.3)  # Bottom 30% as stationary
    
    stationary_periods = rolling_std < motion_threshold
    
    if plot:
        time = np.arange(len(accel_mag)) / sample_rate
        ax.plot(time, accel_mag_no_g, alpha=0.5, label='Acceleration magnitude')
        ax.fill_between(time, -motion_threshold, motion_threshold, 
                       alpha=0.3, color='gray', label='Stationary threshold')
        
        # Highlight stationary periods
        stationary_values = np.where(stationary_periods, accel_mag_no_g, np.nan)
        ax.plot(time[:len(stationary_values)], stationary_values, 
               'r.', markersize=2, label='Stationary')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration Magnitude - g (m/s²)')
        ax.set_title('Motion vs. Stationary Periods')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Cutoff Selection Criteria
    ax = axes[1, 1] if plot else None
    
    if drift_values and plot:
        cutoffs, drifts = zip(*drift_values)
        
        # Normalize drift values
        max_drift = max(drifts)
        normalized_drifts = [d/max_drift for d in drifts]
        
        ax.semilogx(cutoffs, normalized_drifts, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Highpass Cutoff Frequency (Hz)')
        ax.set_ylabel('Normalized Velocity Drift')
        ax.set_title('Drift vs. Cutoff Frequency')
        ax.grid(True, alpha=0.3)
        
        # Find elbow point (where drift reduction diminishes)
        if len(drifts) > 2:
            # Calculate second derivative
            second_diff = np.diff(np.diff(normalized_drifts))
            if len(second_diff) > 0:
                elbow_idx = np.argmax(second_diff) + 1
                recommended_cutoff = cutoffs[elbow_idx]
                ax.axvline(recommended_cutoff, color='red', linestyle='--', 
                          label=f'Recommended: {recommended_cutoff} Hz')
                ax.legend()
                results['recommended_cutoff'] = recommended_cutoff
    
    if plot:
        plt.tight_layout()
        plt.show()
    
    return results

def test_cutoff_frequencies(df, accel_columns, sample_rate=100, 
                           cutoffs=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]):
    """
    Test multiple cutoff frequencies and show their effects.
    """
    fig, axes = plt.subplots(len(cutoffs), 2, figsize=(12, 3*len(cutoffs)))
    if len(cutoffs) == 1:
        axes = axes.reshape(1, -1)
    
    time = np.arange(len(df)) / sample_rate
    
    for i, cutoff in enumerate(cutoffs):
        if cutoff >= sample_rate / 2:
            continue
            
        # Design filter
        b, a = signal.butter(2, cutoff / (sample_rate / 2), btype='high')
        
        # Plot original vs filtered acceleration
        ax = axes[i, 0]
        original = df[accel_columns[0]].values
        filtered = signal.filtfilt(b, a, original)
        
        ax.plot(time, original, 'b-', alpha=0.5, label='Original')
        ax.plot(time, filtered, 'r-', alpha=0.8, label=f'Filtered ({cutoff} Hz)')
        ax.set_ylabel('Accel (m/s²)')
        ax.set_title(f'Highpass Cutoff: {cutoff} Hz')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot integrated velocity
        ax = axes[i, 1]
        dt = 1.0 / sample_rate
        velocity_orig = cumtrapz(original, dx=dt, initial=0)
        velocity_filt = cumtrapz(filtered, dx=dt, initial=0)
        
        ax.plot(time, velocity_orig, 'b-', alpha=0.5, label='From original')
        ax.plot(time, velocity_filt, 'r-', alpha=0.8, label='From filtered')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title(f'Integrated Velocity (drift: {velocity_filt[-1]:.3f} m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

def adaptive_cutoff_selection(df, accel_columns, sample_rate=100):
    """
    Automatically suggest cutoff based on your data characteristics.
    """
    suggestions = {}
    
    # 1. Based on signal power distribution
    for col in accel_columns:
        freqs, psd = signal.welch(df[col].values, fs=sample_rate)
        cumulative_power = np.cumsum(psd) / np.sum(psd)
        
        # Find frequency containing 1% of total power
        idx_1pct = np.where(cumulative_power >= 0.01)[0][0]
        suggestions[f'{col}_1pct_power'] = freqs[idx_1pct]
    
    # 2. Based on motion characteristics
    accel_mag = np.sqrt(sum(df[col]**2 for col in accel_columns))
    
    # Estimate stationary noise level
    rolling_std = pd.Series(accel_mag).rolling(int(sample_rate)).std()
    noise_level = rolling_std.quantile(0.1)  # Bottom 10% as noise floor
    
    # Suggest cutoff based on noise
    if noise_level < 0.05:  # Low noise
        suggestions['noise_based'] = 0.05
    elif noise_level < 0.1:  # Medium noise
        suggestions['noise_based'] = 0.1
    else:  # High noise
        suggestions['noise_based'] = 0.2
    
    # 3. Based on application type (heuristic)
    print("\n=== ADAPTIVE CUTOFF SUGGESTIONS ===")
    print(f"Based on power distribution: {np.mean(list(suggestions.values())):.3f} Hz")
    print(f"Based on noise level: {suggestions['noise_based']} Hz")
    
    # Overall recommendation
    all_suggestions = list(suggestions.values())
    recommended = np.median(all_suggestions)
    
    print(f"\nRECOMMENDED CUTOFF: {recommended:.3f} Hz")
    
    # Application-specific guidance
    print("\n=== APPLICATION-SPECIFIC GUIDELINES ===")
    print("- Slow movements (walking): 0.1 - 0.2 Hz")
    print("- Normal activities: 0.1 - 0.5 Hz")
    print("- Including gestures: 0.05 - 0.1 Hz")
    print("- Vibration analysis: 0.5 - 1.0 Hz")
    
    return recommended

# Practical rule-of-thumb function
def quick_cutoff_selection(activity_type='general'):
    """
    Quick selection based on activity type.
    
    Args:
        activity_type: 'slow', 'general', 'sports', 'vibration'
    
    Returns:
        Recommended cutoff frequency
    """
    cutoffs = {
        'slow': 0.1,      # Slow movements, walking
        'general': 0.2,   # General daily activities  
        'sports': 0.5,    # Sports, running, quick movements
        'vibration': 1.0, # Vibration analysis
        'precision': 0.05 # When you need to preserve slow movements
    }
    
    return cutoffs.get(activity_type, 0.1)

# Example usage
if __name__ == "__main__":
    print("=== HIGHPASS CUTOFF SELECTION GUIDE ===\n")
    
    # Quick selection
    activity = 'general'
    cutoff = quick_cutoff_selection(activity)
    print(f"For {activity} activities, use cutoff: {cutoff} Hz")

    # Example DataFrame
    df = pd.read_csv('acceleration_with_times.csv')  # Load your data here
    
    # If you have data:
    results = analyze_frequency_for_cutoff(df, sample_rate=100)
    recommended = adaptive_cutoff_selection(df, ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'])
    
    print("\n=== KEY PRINCIPLES ===")
    print("• Lower cutoff (0.01-0.1 Hz): Preserves more motion, more drift")
    print("• Higher cutoff (0.2-1.0 Hz): Less drift, may lose slow movements")
    print("• Start with 0.1 Hz for most applications")
    print("• Increase if you see excessive drift")
    print("• Decrease if you're losing important slow motions")