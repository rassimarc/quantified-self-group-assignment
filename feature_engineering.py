import pandas as pd
import numpy as np
from scipy import signal

import kalman_filter as kf
import add_times_to_data as atd
import plots
from get_sampling_rate import calculate_sampling_frequency

from scipy import signal

from scipy.ndimage import label

def remove_gravity_and_calculate_velocity(accel_df, 
                                        gravity_magnitude=9.81,
                                        initial_velocity=(0, 0, 0),
                                        highpass_cutoff=0.05,
                                        sample_rate=99.4):
    """
    Calculate velocity with gravity removal and high-pass filtering to reduce drift.
    
    Args:
        accel_df: DataFrame with acceleration data
        gravity_magnitude: Expected gravity magnitude (default 9.81 m/s²)
        initial_velocity: Initial velocity tuple
        highpass_cutoff: High-pass filter cutoff frequency (Hz)
        sample_rate: Sampling rate (Hz)
    """
    df = accel_df.copy()
    
    # Step 1: Estimate and remove gravity using stationary periods
    # Calculate the magnitude of acceleration
    df['accel_magnitude'] = np.sqrt(
        df['X (m/s^2)']**2 + df['Y (m/s^2)']**2 + df['Z (m/s^2)']**2
    )

    # Find stationary periods: where acceleration changes very little for a window
    window_size = 500  # samples (adjust as needed)
    std_threshold = 0.2 # m/s^2, adjust for your sensor noise

    # Rolling standard deviation of acceleration magnitude
    df['accel_std'] = df['accel_magnitude'].rolling(window=window_size, center=True).std()

    # Stationary mask: low std means stationary
    stationary_mask = df['accel_std'] < std_threshold

    # Only consider long enough stationary periods

    labeled, num_features = label(stationary_mask)
    stationary_indices = []
    min_stationary_length = 20  # samples

    for i in range(1, num_features + 1):
        idx = np.where(labeled == i)[0]
        if len(idx) >= min_stationary_length:
            stationary_indices.extend(idx)

    # Show at which time the stationary samples were found
    if len(stationary_indices) > 0:
        stationary_times = df['Time (s)'].iloc[stationary_indices]
        print(f"Stationary samples found at times: {stationary_times.values}")
    else:
        print("No stationary samples found.")

    if stationary_indices:
        # Use all stationary samples to estimate gravity
        gravity_x = df['X (m/s^2)'].iloc[stationary_indices].mean()
        gravity_y = df['Y (m/s^2)'].iloc[stationary_indices].mean()
        gravity_z = df['Z (m/s^2)'].iloc[stationary_indices].mean()
    else:
        # Fallback: use initial samples
        initial_samples = min(50, len(df) // 10)
        gravity_x = df['X (m/s^2)'].iloc[:initial_samples].mean()
        gravity_y = df['Y (m/s^2)'].iloc[:initial_samples].mean()
        gravity_z = df['Z (m/s^2)'].iloc[:initial_samples].mean()

    df['X_no_gravity'] = df['X (m/s^2)'] - gravity_x
    df['Y_no_gravity'] = df['Y (m/s^2)'] - gravity_y
    df['Z_no_gravity'] = df['Z (m/s^2)'] - gravity_z

    # Print the time where gravity was estimated
    print(f"Estimated gravity at time {df['Time (s)'].iloc[0]}: {gravity_x:.2f} m/s², {gravity_y:.2f} m/s², {gravity_z:.2f} m/s²")
    print(f"Gravity magnitude: {np.sqrt(gravity_x**2 + gravity_y**2 + gravity_z**2):.2f} m/s²")

    # Step 2: Apply high-pass filter to remove low-frequency drift
    # Design butterworth high-pass filter
    nyquist = sample_rate / 2
    if highpass_cutoff < nyquist:
        b, a = signal.butter(2, highpass_cutoff / nyquist, btype='high')
        
        # Apply filter (with padding to reduce edge effects)
        df['X_filtered'] = signal.filtfilt(b, a, df['X_no_gravity'])
        df['Y_filtered'] = signal.filtfilt(b, a, df['Y_no_gravity'])
        df['Z_filtered'] = signal.filtfilt(b, a, df['Z_no_gravity'])
    else:
        # If cutoff is too high, just use gravity-removed data
        df['X_filtered'] = df['X_no_gravity']
        df['Y_filtered'] = df['Y_no_gravity']
        df['Z_filtered'] = df['Z_no_gravity']
    
    # Apply Kalman filter to further reduce noise
    df['X_filtered'] = kf.kalman_filter_1d(df['X_filtered'].values)
    df['Y_filtered'] = kf.kalman_filter_1d(df['Y_filtered'].values)
    df['Z_filtered'] = kf.kalman_filter_1d(df['Z_filtered'].values)

    # Step 3: Integrate filtered acceleration
    df['dt'] = df['Time (s)'].diff()
    if pd.isna(df['dt'].iloc[0]) and len(df) > 1:
        df.loc[0, 'dt'] = df['Time (s)'].iloc[1] - df['Time (s)'].iloc[0]

    # Initialize velocity arrays
    vx = np.zeros(len(df))
    vy = np.zeros(len(df))
    vz = np.zeros(len(df))
    
    vx[0], vy[0], vz[0] = initial_velocity

    # Integrate using trapezoidal rule, reset velocity to zero during stationary periods
    for i in range(1, len(df)):
        dt = df['dt'].iloc[i]
        if i in stationary_indices:
            vx[i] = 0
            vy[i] = 0
            vz[i] = 0
        else:
            vx[i] = vx[i-1] + (df['X_filtered'].iloc[i] + df['X_filtered'].iloc[i-1]) / 2 * dt
            vy[i] = vy[i-1] + (df['Y_filtered'].iloc[i] + df['Y_filtered'].iloc[i-1]) / 2 * dt
            vz[i] = vz[i-1] + (df['Z_filtered'].iloc[i] + df['Z_filtered'].iloc[i-1]) / 2 * dt
    
    df['Vx (m/s)'] = vx
    df['Vy (m/s)'] = vy
    df['Vz (m/s)'] = vz
    df['V_magnitude (m/s)'] = np.sqrt(vx**2 + vy**2 + vz**2)

    # Velocity magnitude after filtering
    df['V_magnitude'] = np.sqrt(df['Vx (m/s)']**2 + df['Vy (m/s)']**2 + df['Vz (m/s)']**2)

    return df

def clean_and_filter_gyroscope_data(gyro_df, sample_rate=99.4):
    """
    Clean and filter gyroscope data using similar techniques as for acceleration.
    
    Args:
        gyro_df: DataFrame with gyroscope data
        sample_rate: Sampling rate (Hz)

    Returns:
        DataFrame with cleaned and filtered gyroscope data
    """
    # Remove NaN values
    gyro_df = gyro_df.dropna()

    # Step 1: Apply high-pass filter to remove low-frequency drift
    nyquist = sample_rate / 2
    highpass_cutoff = 0.05  # High-pass filter cutoff frequency (Hz)
    if highpass_cutoff < nyquist:
        b, a = signal.butter(2, highpass_cutoff / nyquist, btype='high')
        
        # Apply filter (with padding to reduce edge effects)
        gyro_df['X_gyro_filtered'] = signal.filtfilt(b, a, gyro_df['X_gyro (rad/s)'])
        gyro_df['Y_gyro_filtered'] = signal.filtfilt(b, a, gyro_df['Y_gyro (rad/s)'])
        gyro_df['Z_gyro_filtered'] = signal.filtfilt(b, a, gyro_df['Z_gyro (rad/s)'])
    else:
        # If cutoff is too high, just use raw gyroscope data
        gyro_df['X_gyro_filtered'] = gyro_df['X_gyro (rad/s)']
        gyro_df['Y_gyro_filtered'] = gyro_df['Y_gyro (rad/s)']
        gyro_df['Z_gyro_filtered'] = gyro_df['Z_gyro (rad/s)']

    # Step 2: Apply Kalman filter to further reduce noise
    gyro_df['X_gyro_filtered'] = kf.kalman_filter_1d(gyro_df['X_gyro_filtered'].values)
    gyro_df['Y_gyro_filtered'] = kf.kalman_filter_1d(gyro_df['Y_gyro_filtered'].values)
    gyro_df['Z_gyro_filtered'] = kf.kalman_filter_1d(gyro_df['Z_gyro_filtered'].values)

    # Step 3: Calculate angular velocity magnitude
    gyro_df['gyro_magnitude'] = np.sqrt(
        gyro_df['X_gyro_filtered']**2 + gyro_df['Y_gyro_filtered']**2 + gyro_df['Z_gyro_filtered']**2
    )

    return gyro_df

# Example usage:
if __name__ == "__main__":

    """ Uncomment to add system times and gyroscope data to acceleration data.
    """
    # Add paths to your files
    acceleration_file = "data/Linear Acceleration.csv"
    gyroscope_file = "data/Gyroscope.csv"
    events_file = "data/meta/time.csv"

    # Get acceleration data with system times
    result = atd.get_acceleration_times_and_gyroscope_data(
        acceleration_file, gyroscope_file, events_file
    )

    # If you want to save the result to a CSV file
    # result.to_csv("bike1.csv", index=False)
    
    # # Replace with your actual file paths
    # acceleration_file = "bike1.csv"
    # data = pd.read_csv(acceleration_file)

    data = result

    # Display the result
    print("Acceleration data with corresponding system times:")
    print(data.head(10))

    # Calculate sampling frequency
    sampling_results = calculate_sampling_frequency(acceleration_file)
    print(f"Sampling Frequency: {sampling_results['sampling_frequency_hz']:.2f} Hz")

    df_filtered = remove_gravity_and_calculate_velocity(data, sample_rate=sampling_results['sampling_frequency_hz'])
    df_filtered = clean_and_filter_gyroscope_data(df_filtered, sample_rate=sampling_results['sampling_frequency_hz'])

    print(f"Max velocity (uncorrected): {df_filtered['V_magnitude (m/s)'].max():.2f} m/s")

    plots.plot_acceleration(df_filtered)
    plots.plot_gyroscope(df_filtered)

    # Save to a new CSV file
    df_filtered.to_csv("clean_data/walk/xiyan_walk7.csv", index=False)
    print("\nResult saved to 'clean_data/walk/xiyan_walk2.csv'")