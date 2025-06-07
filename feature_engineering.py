import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


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
                                        highpass_cutoff=0.1,
                                        sample_rate=100):
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
    
    # Step 1: Estimate and remove gravity (using initial samples)
    # Assuming device is initially stationary
    initial_samples = min(50, len(df) // 10)  # Use first 50 samples or 10% of data
    
    # Average acceleration over initial period (assumed stationary)
    gravity_x = df['X (m/s^2)'].iloc[:initial_samples].mean()
    gravity_y = df['Y (m/s^2)'].iloc[:initial_samples].mean()
    gravity_z = df['Z (m/s^2)'].iloc[:initial_samples].mean()
    
    # Remove estimated gravity
    df['X_no_gravity'] = df['X (m/s^2)'] - gravity_x
    df['Y_no_gravity'] = df['Y (m/s^2)'] - gravity_y
    df['Z_no_gravity'] = df['Z (m/s^2)'] - gravity_z
    
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
    
    # Step 3: Integrate filtered acceleration
    df['dt'] = df['Time (s)'].diff()
    if pd.isna(df['dt'].iloc[0]) and len(df) > 1:
        df.loc[0, 'dt'] = df['Time (s)'].iloc[1] - df['Time (s)'].iloc[0]
    
    # Initialize velocity arrays
    vx = np.zeros(len(df))
    vy = np.zeros(len(df))
    vz = np.zeros(len(df))
    
    vx[0], vy[0], vz[0] = initial_velocity
    
    # Integrate using trapezoidal rule
    for i in range(1, len(df)):
        dt = df['dt'].iloc[i]
        vx[i] = vx[i-1] + (df['X_filtered'].iloc[i] + df['X_filtered'].iloc[i-1]) / 2 * dt
        vy[i] = vy[i-1] + (df['Y_filtered'].iloc[i] + df['Y_filtered'].iloc[i-1]) / 2 * dt
        vz[i] = vz[i-1] + (df['Z_filtered'].iloc[i] + df['Z_filtered'].iloc[i-1]) / 2 * dt
    
    df['Vx (m/s)'] = vx
    df['Vy (m/s)'] = vy
    df['Vz (m/s)'] = vz
    df['V_magnitude (m/s)'] = np.sqrt(vx**2 + vy**2 + vz**2)
    
    return df

def apply_zupt(accel_df, velocity_df, motion_threshold=0.5, window_size=10):
    """
    Apply Zero-velocity Update (ZUPT) to correct for drift during stationary periods.
    
    Args:
        accel_df: Original acceleration data
        velocity_df: DataFrame with calculated velocities
        motion_threshold: Acceleration magnitude threshold for detecting motion
        window_size: Window size for detecting stationary periods
    """
    df = velocity_df.copy()
    
    # Calculate acceleration magnitude
    accel_magnitude = np.sqrt(
        accel_df['X (m/s^2)']**2 + 
        accel_df['Y (m/s^2)']**2 + 
        accel_df['Z (m/s^2)']**2
    )
    
    # Remove gravity from magnitude (approximately)
    accel_magnitude_no_g = np.abs(accel_magnitude - 9.81)
    
    # Detect stationary periods (low acceleration variance)
    stationary = np.zeros(len(df), dtype=bool)
    
    for i in range(window_size, len(df) - window_size):
        window = accel_magnitude_no_g[i-window_size:i+window_size]
        if np.std(window) < motion_threshold and np.mean(window) < motion_threshold:
            stationary[i] = True
    
    # Apply ZUPT corrections
    df['ZUPT_applied'] = stationary
    df['Vx_corrected'] = df['Vx (m/s)'].copy()
    df['Vy_corrected'] = df['Vy (m/s)'].copy()
    df['Vz_corrected'] = df['Vz (m/s)'].copy()
    
    # Reset velocity to zero during stationary periods
    df.loc[stationary, 'Vx_corrected'] = 0
    df.loc[stationary, 'Vy_corrected'] = 0
    df.loc[stationary, 'Vz_corrected'] = 0
    
    # Interpolate between ZUPT points to smooth transitions
    for col in ['Vx_corrected', 'Vy_corrected', 'Vz_corrected']:
        # Find ZUPT points
        zupt_indices = np.where(stationary)[0]
        if len(zupt_indices) > 1:
            # Linear interpolation between ZUPT points
            for i in range(len(zupt_indices) - 1):
                start_idx = zupt_indices[i]
                end_idx = zupt_indices[i + 1]
                if end_idx - start_idx > 1:
                    # Interpolate
                    df.loc[start_idx:end_idx, col] = np.linspace(
                        df.loc[start_idx, col],
                        df.loc[end_idx, col],
                        end_idx - start_idx + 1
                    )
    
    df['V_magnitude_corrected'] = np.sqrt(
        df['Vx_corrected']**2 + 
        df['Vy_corrected']**2 + 
        df['Vz_corrected']**2
    )
    
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
    if 'X_filtered' in df.columns:
        ax = axes[1]
        ax.plot(df['Time (s)'], df['X_filtered'], 'r-', alpha=0.5, label='X filtered')
        ax.plot(df['Time (s)'], df['Y_filtered'], 'g-', alpha=0.5, label='Y filtered')
        ax.plot(df['Time (s)'], df['Z_filtered'], 'b-', alpha=0.5, label='Z filtered')
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
    
    # Plot corrected velocity if available
    if 'Vx_corrected' in df.columns:
        ax = axes[3]
        ax.plot(df['Time (s)'], df['Vx_corrected'], 'r-', alpha=0.7, label='Vx corrected')
        ax.plot(df['Time (s)'], df['Vy_corrected'], 'g-', alpha=0.7, label='Vy corrected')
        ax.plot(df['Time (s)'], df['Vz_corrected'], 'b-', alpha=0.7, label='Vz corrected')
        
        # Mark ZUPT points
        zupt_times = df[df['ZUPT_applied']]['Time (s)']
        for t in zupt_times[::10]:  # Plot every 10th point to avoid crowding
            ax.axvline(x=t, color='gray', alpha=0.2, linewidth=0.5)
        
        ax.set_ylabel('Corrected Velocity (m/s)')
        ax.set_xlabel('Time (s)')
        ax.set_title('ZUPT-Corrected Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Replace with your actual file paths
    acceleration_file = "data\\Measurement 6425 1240PM walking and standing 2025-06-06 18-43-57\\Linear Accelerometer.csv"
    events_file = "data\\Measurement 6425 1240PM walking and standing 2025-06-06 18-43-57\\meta\\time.csv"

    # Get the acceleration data with corresponding times
    result = get_acceleration_times(acceleration_file, events_file)
    
    # Display the result
    print("Acceleration data with corresponding system times:")
    print(result.head(10))

    df_filtered = remove_gravity_and_calculate_velocity(result)

    df_corrected = apply_zupt(result, df_filtered)

    print(f"Max velocity (uncorrected): {df_filtered['V_magnitude (m/s)'].max():.2f} m/s")
    if 'V_magnitude_corrected' in df_corrected.columns:
        print(f"Max velocity (ZUPT corrected): {df_corrected['V_magnitude_corrected'].max():.2f} m/s")

    analyze_drift(df_corrected)

    # Save to a new CSV file
    df_corrected.to_csv("acceleration_with_times.csv", index=False)
    print("\nResult saved to 'acceleration_with_times.csv'")