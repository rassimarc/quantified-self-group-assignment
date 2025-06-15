import matplotlib.pyplot as plt

def plot_acceleration(df):
    """
    Analyze and visualize drift in the velocity calculations.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    # Plot raw acceleration
    ax = axes[0]
    # ax.plot(df['Time (s)'], df['X_filtered'], 'r-', alpha=0.5, label='X filtered')
    # ax.plot(df['Time (s)'], df['Y_filtered'], 'g-', alpha=0.5, label='Y filtered')
    # ax.plot(df['Time (s)'], df['Z_filtered'], 'b-', alpha=0.5, label='Z filtered')
    # ax.set_ylabel('Filtered Acceleration (m/s²)')
    # ax.set_title('High-pass Filtered Acceleration (Gravity Removed)')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    ax.plot(df['Time (s)'], df['X (m/s^2)'], 'r-', alpha=0.5, label='X')
    ax.plot(df['Time (s)'], df['Y (m/s^2)'], 'g-', alpha=0.5, label='Y')
    ax.plot(df['Time (s)'], df['Z (m/s^2)'], 'b-', alpha=0.5, label='Z')
    ax.set_ylabel('Raw Acceleration (m/s²)')
    ax.set_title('Raw Accelerometer Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot filtered acceleration
    # if 'X_no_gravity' in df.columns:
    #     ax = axes[1]
    #     ax.plot(df['Time (s)'], df['X_no_gravity'], 'r-', alpha=0.5, label='X no_gravity')
    #     ax.plot(df['Time (s)'], df['Y_no_gravity'], 'g-', alpha=0.5, label='Y no_gravity')
    #     ax.plot(df['Time (s)'], df['Z_no_gravity'], 'b-', alpha=0.5, label='Z no_gravity')
    #     ax.set_ylabel('Filtered Acceleration (m/s²)')
    #     ax.set_title('High-pass Filtered Acceleration (Gravity Removed)')
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)
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

def plot_gyroscope(df):
    """
    Plot gyroscope data.
    
    Args:
        df: DataFrame with gyroscope data
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    ax = axes[0]
    ax.plot(df['Time (s)'], df['X_gyro (rad/s)'], 'r-', alpha=0.5, label='X Gyro')
    ax.plot(df['Time (s)'], df['Y_gyro (rad/s)'], 'g-', alpha=0.5, label='Y Gyro')
    ax.plot(df['Time (s)'], df['Z_gyro (rad/s)'], 'b-', alpha=0.5, label='Z Gyro')
    ax.set_ylabel('Gyroscope Data (rad/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Gyroscope Data Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(df['Time (s)'], df['X_gyro_filtered'], 'r-', alpha=0.5, label='X Gyro')
    ax.plot(df['Time (s)'], df['Y_gyro_filtered'], 'g-', alpha=0.5, label='Y Gyro')
    ax.plot(df['Time (s)'], df['Z_gyro_filtered'], 'b-', alpha=0.5, label='Z Gyro')
    ax.set_ylabel('Gyroscope Data (rad/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Gyroscope Filtered Data Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()