import pandas as pd
from datetime import timedelta
import numpy as np
from scipy import interpolate

def get_acceleration_times_and_gyroscope_data(acceleration_file, gyroscope_file, events_file):
    """
    Maps experiment times from acceleration data to actual system times
    based on start/pause events.
    
    Args:
        acceleration_file: Path to CSV file with acceleration data
        gyroscope_file: Path to CSV file with gyroscope data
        events_file: Path to CSV file with start/pause events

    Returns:
        DataFrame with acceleration data and corresponding system times
    """
    
    # Read the data files
    accel_df = pd.read_csv(acceleration_file)
    gyro_df = pd.read_csv(gyroscope_file)
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
    
    gyro_df.columns = ['Time (s)', 'X_gyro (rad/s)', 'Y_gyro (rad/s)', 'Z_gyro (rad/s)']

    interp_x = interpolate.interp1d(gyro_df['Time (s)'], gyro_df['X_gyro (rad/s)'], 
                                kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_y = interpolate.interp1d(gyro_df['Time (s)'], gyro_df['Y_gyro (rad/s)'], 
                                kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_z = interpolate.interp1d(gyro_df['Time (s)'], gyro_df['Z_gyro (rad/s)'], 
                                kind='linear', bounds_error=False, fill_value='extrapolate')
    result_df['X_gyro (rad/s)'] = interp_x(result_df['Time (s)'])
    result_df['Y_gyro (rad/s)'] = interp_y(result_df['Time (s)'])
    result_df['Z_gyro (rad/s)'] = interp_z(result_df['Time (s)'])
    
    return result_df