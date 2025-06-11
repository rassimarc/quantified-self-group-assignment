import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def calculate_velocity_from_acceleration(accel_df, initial_velocity=(0, 0, 0)):
    """
    Calculate velocity from acceleration data using numerical integration.
    
    Args:
        accel_df: DataFrame with columns 'Time (s)', 'X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'
        initial_velocity: Tuple of (vx0, vy0, vz0) initial velocities in m/s
    
    Returns:
        DataFrame with velocity columns added
    """
    # Create a copy of the dataframe
    df = accel_df.copy()
    
    # Calculate time differences between measurements
    df['dt'] = df['Time (s)'].diff()
    # For the first row, use the difference between first and second row
    if pd.isna(df['dt'].iloc[0]) and len(df) > 1:
        df.loc[0, 'dt'] = df['Time (s)'].iloc[1] - df['Time (s)'].iloc[0]
    
    # Initialize velocity arrays
    vx = np.zeros(len(df))
    vy = np.zeros(len(df))
    vz = np.zeros(len(df))
    
    # Set initial velocities
    vx[0], vy[0], vz[0] = initial_velocity
    
    # Integrate acceleration to get velocity using trapezoidal rule
    for i in range(1, len(df)):
        dt = df['dt'].iloc[i]
        
        # Trapezoidal integration: v[i] = v[i-1] + (a[i] + a[i-1])/2 * dt
        vx[i] = vx[i-1] + (df['X (m/s^2)'].iloc[i] + df['X (m/s^2)'].iloc[i-1]) / 2 * dt
        vy[i] = vy[i-1] + (df['Y (m/s^2)'].iloc[i] + df['Y (m/s^2)'].iloc[i-1]) / 2 * dt
        vz[i] = vz[i-1] + (df['Z (m/s^2)'].iloc[i] + df['Z (m/s^2)'].iloc[i-1]) / 2 * dt
    
    # Add velocity columns to dataframe
    df['Vx (m/s)'] = vx
    df['Vy (m/s)'] = vy
    df['Vz (m/s)'] = vz
    
    # Calculate magnitude of velocity
    df['V_magnitude (m/s)'] = np.sqrt(vx**2 + vy**2 + vz**2)
    
    return df

# def get_velocity_at_time(df, target_time, time_column='Time (s)'):
    """
    Get velocity at a specific time point using interpolation if necessary.
    
    Args:
        df: DataFrame with velocity data
        target_time: The time at which to find velocity
        time_column: Name of the time column
    
    Returns:
        Dictionary with velocity components and magnitude at the target time
    """
    # Check if exact time exists
    exact_match = df[df[time_column] == target_time]
    
    if not exact_match.empty:
        row = exact_match.iloc[0]
        return {
            'time': target_time,
            'vx': row['Vx (m/s)'],
            'vy': row['Vy (m/s)'],
            'vz': row['Vz (m/s)'],
            'v_magnitude': row['V_magnitude (m/s)']
        }
    
    # If no exact match, interpolate
    # Find the closest times before and after
    before = df[df[time_column] <= target_time]
    after = df[df[time_column] >= target_time]
    
    if before.empty or after.empty:
        return None
    
    # Get the closest points
    t1 = before[time_column].iloc[-1]
    t2 = after[time_column].iloc[0]
    
    if t1 == t2:  # Edge case
        row = before.iloc[-1]
        return {
            'time': target_time,
            'vx': row['Vx (m/s)'],
            'vy': row['Vy (m/s)'],
            'vz': row['Vz (m/s)'],
            'v_magnitude': row['V_magnitude (m/s)']
        }
    
    # Linear interpolation
    row1 = before.iloc[-1]
    row2 = after.iloc[0]
    
    # Interpolation factor
    factor = (target_time - t1) / (t2 - t1)
    
    # Interpolate each component
    vx_interp = row1['Vx (m/s)'] + factor * (row2['Vx (m/s)'] - row1['Vx (m/s)'])
    vy_interp = row1['Vy (m/s)'] + factor * (row2['Vy (m/s)'] - row1['Vy (m/s)'])
    vz_interp = row1['Vz (m/s)'] + factor * (row2['Vz (m/s)'] - row1['Vz (m/s)'])
    v_mag_interp = np.sqrt(vx_interp**2 + vy_interp**2 + vz_interp**2)
    
    return {
        'time': target_time,
        'vx': vx_interp,
        'vy': vy_interp,
        'vz': vz_interp,
        'v_magnitude': v_mag_interp
    }

def plot_velocity_and_acceleration(df):
    """
    Plot acceleration and velocity over time.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot acceleration
    ax1.plot(df['Time (s)'], df['X (m/s^2)'], 'r-', label='X', alpha=0.7)
    ax1.plot(df['Time (s)'], df['Y (m/s^2)'], 'g-', label='Y', alpha=0.7)
    ax1.plot(df['Time (s)'], df['Z (m/s^2)'], 'b-', label='Z', alpha=0.7)
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Acceleration over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot velocity
    ax2.plot(df['Time (s)'], df['Vx (m/s)'], 'r-', label='Vx', alpha=0.7)
    ax2.plot(df['Time (s)'], df['Vy (m/s)'], 'g-', label='Vy', alpha=0.7)
    ax2.plot(df['Time (s)'], df['Vz (m/s)'], 'b-', label='Vz', alpha=0.7)
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot velocity magnitude
    ax3.plot(df['Time (s)'], df['V_magnitude (m/s)'], 'purple', label='|V|', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity Magnitude (m/s)')
    ax3.set_title('Velocity Magnitude over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
