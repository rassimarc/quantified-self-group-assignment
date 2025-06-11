import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import with_gravity as wg

from scipy.ndimage import label

def get_acceleration_times(acceleration_file, events_file):
    """
    Maps experiment times from acceleration data to actual system times
    based on start/pause events.
    
    Args:
        acceleration_file: Path to CSV file with acceleration data
        events_file: Path to CSV file with start/pause events
    
    Returns:
        DataFrame with acceleration data and corresponding system times
    """
    
    # Read the data files
    accel_df = pd.read_csv(acceleration_file)
    events_df = pd.read_csv(events_file)
    
    # Convert system time to datetime objects
    events_df['datetime'] = pd.to_datetime(events_df['system time text'])
    
    # Create a list to store the calculated times
    calculated_times = []
    
    # Get start events and their corresponding system times
    start_events = events_df[events_df['event'] == 'START'].copy()
    start_events = start_events.sort_values('experiment time')
    
    for _, accel_row in accel_df.iterrows():
        experiment_time = accel_row['Time (s)']
        
        # Find the most recent START event before or at this experiment time
        valid_starts = start_events[start_events['experiment time'] <= experiment_time]
        
        if len(valid_starts) > 0:
            # Get the most recent start event
            latest_start = valid_starts.iloc[-1]
            
            # Calculate the offset from the start time
            time_offset = experiment_time - latest_start['experiment time']
            
            # Add the offset to the system time of the start event
            system_datetime = latest_start['datetime'] + timedelta(seconds=time_offset)
            calculated_times.append(system_datetime)
        else:
            # If no start event found, use None or handle as needed
            calculated_times.append(None)
    
    # Add the calculated times to the acceleration dataframe
    result_df = accel_df.copy()
    result_df['System Time'] = calculated_times
    # result_df['System Time Text'] = [dt.strftime('%Y-%m-%d %H:%M:%S.%f UTC+02:00') if dt else None 
    #                                 for dt in calculated_times]
    
    return result_df






def remove_gravity_and_calculate_velocity(accel_df, 
                                        gravity_magnitude=9.81,
                                        initial_velocity=(0, 0, 0),
                                        highpass_cutoff=0.391,
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
    window_size = 100  # samples (adjust as needed)
    std_threshold = 0.01  # m/s^2, adjust for your sensor noise

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

    # Remove estimated gravity
    df['X_no_gravity'] = df['X (m/s^2)'] - gravity_x
    df['Y_no_gravity'] = df['Y (m/s^2)'] - gravity_y
    df['Z_no_gravity'] = df['Z (m/s^2)'] - gravity_z

    # Print the time where gravity was estimated
    # print(f"Estimated gravity at time {df['Time (s)'].iloc[0]}: {gravity_x:.2f} m/s², {gravity_y:.2f} m/s², {gravity_z:.2f} m/s²")
    # print(f"Gravity magnitude: {np.sqrt(gravity_x**2 + gravity_y**2 + gravity_z**2):.2f} m/s²")


    # # Step 2: Apply high-pass filter to remove low-frequency drift
    # # Design butterworth high-pass filter
    # nyquist = sample_rate / 2
    # if highpass_cutoff < nyquist:
    #     b, a = signal.butter(2, highpass_cutoff / nyquist, btype='high')
        
    #     # Apply filter (with padding to reduce edge effects)
    #     df['X_filtered'] = signal.filtfilt(b, a, df['X_no_gravity'])
    #     df['Y_filtered'] = signal.filtfilt(b, a, df['Y_no_gravity'])
    #     df['Z_filtered'] = signal.filtfilt(b, a, df['Z_no_gravity'])
    # else:
    #     # If cutoff is too high, just use gravity-removed data
    #     df['X_filtered'] = df['X_no_gravity']
    #     df['Y_filtered'] = df['Y_no_gravity']
    #     df['Z_filtered'] = df['Z_no_gravity']
    
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
            vx[i] = vx[i-1] + (df['X_no_gravity'].iloc[i] + df['X_no_gravity'].iloc[i-1]) / 2 * dt
            vy[i] = vy[i-1] + (df['Y_no_gravity'].iloc[i] + df['Y_no_gravity'].iloc[i-1]) / 2 * dt
            vz[i] = vz[i-1] + (df['Z_no_gravity'].iloc[i] + df['Z_no_gravity'].iloc[i-1]) / 2 * dt
        # vx[i] = vx[i-1] + (df['X_filtered'].iloc[i] + df['X_filtered'].iloc[i-1]) / 2 * dt
        # vy[i] = vy[i-1] + (df['Y_filtered'].iloc[i] + df['Y_filtered'].iloc[i-1]) / 2 * dt
        # vz[i] = vz[i-1] + (df['Z_filtered'].iloc[i] + df['Z_filtered'].iloc[i-1]) / 2 * dt
    
    df['Vx (m/s)'] = vx
    df['Vy (m/s)'] = vy
    df['Vz (m/s)'] = vz
    df['V_magnitude (m/s)'] = np.sqrt(vx**2 + vy**2 + vz**2)

    # # Add columns for the difference between filtered and unfiltered acceleration
    # df['X_diff'] = df['X_no_gravity'] - df['X_filtered']  
    # df['Y_diff'] = df['Y_no_gravity'] - df['Y_filtered']
    # df['Z_diff'] = df['Z_no_gravity'] - df['Z_filtered']
    # print("\nDifference between unfiltered and filtered acceleration:")
    # print(f"X mean diff: {df['X_diff'].mean():.3f} m/s²")
    # print(f"Y mean diff: {df['Y_diff'].mean():.3f} m/s²") 
    # print(f"Z mean diff: {df['Z_diff'].mean():.3f} m/s²")

    # Velocity magnitude after filtering
    df['V_magnitude'] = np.sqrt(df['Vx (m/s)']**2 + df['Vy (m/s)']**2 + df['Vz (m/s)']**2)

    return df

def analyze_drift(df):
    """
    Analyze and visualize drift in the velocity calculations.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    # Plot raw acceleration
    ax = axes[0]
    ax.plot(df['Time (s)'], df['X (m/s^2)'], 'r-', alpha=0.5, label='X')
    ax.plot(df['Time (s)'], df['Y (m/s^2)'], 'g-', alpha=0.5, label='Y')
    ax.plot(df['Time (s)'], df['Z (m/s^2)'], 'b-', alpha=0.5, label='Z')
    ax.set_ylabel('Raw Acceleration (m/s²)')
    ax.set_title('Raw Accelerometer Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot filtered acceleration
    if 'X_no_gravity' in df.columns:
        ax = axes[1]
        ax.plot(df['Time (s)'], df['X_no_gravity'], 'r-', alpha=0.5, label='X no_gravity')
        ax.plot(df['Time (s)'], df['Y_no_gravity'], 'g-', alpha=0.5, label='Y no_gravity')
        ax.plot(df['Time (s)'], df['Z_no_gravity'], 'b-', alpha=0.5, label='Z no_gravity')
        ax.set_ylabel('Filtered Acceleration (m/s²)')
        ax.set_title('High-pass Filtered Acceleration (Gravity Removed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot uncorrected velocity
    ax = axes[2]
    ax.plot(df['Time (s)'], df['Vx (m/s)'], 'r-', alpha=0.5, label='Vx')
    ax.plot(df['Time (s)'], df['Vy (m/s)'], 'g-', alpha=0.5, label='Vy')
    ax.plot(df['Time (s)'], df['Vz (m/s)'], 'b-', alpha=0.5, label='Vz')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Integrated Velocity (with drift)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot velocity magnitude
    ax = axes[3]
    ax.plot(df['Time (s)'], df['V_magnitude (m/s)'], 'purple', label='|V|', linewidth=2)
    ax.set_ylabel('Velocity Magnitude (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Velocity Magnitude (with drift)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Replace with your actual file paths
    acceleration_file = "acceleration_with_times.csv"

    data = pd.read_csv(acceleration_file)

    # test = wg.calculate_velocity_from_acceleration(result)
    # wg.plot_velocity_and_acceleration(test)
    
    # Display the result
    print("Acceleration data with corresponding system times:")
    print(data.head(10))

    df_filtered = remove_gravity_and_calculate_velocity(data)

    print(f"Max velocity (uncorrected): {df_filtered['V_magnitude (m/s)'].max():.2f} m/s")

    analyze_drift(df_filtered)

    # Save to a new CSV file
    df_filtered.to_csv("clean_acceleration.csv", index=False)
    print("\nResult saved to 'clean_acceleration.csv'")