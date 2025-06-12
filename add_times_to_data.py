import pandas as pd
from datetime import timedelta
import numpy as np

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
    
    # result_df = merge_gyroscope_data_with_acceleration(result_df, gyro_df)

    return result_df

def merge_gyroscope_data_with_acceleration(accel_df, gyro_df, time_tolerance=0.03):
    """
    Merges gyroscope data with acceleration data based on system time.
    
    Args:
        accel_df: DataFrame with acceleration data and system times
        gyro_df: DataFrame with gyroscope data

    Returns:
        Merged DataFrame with both acceleration and gyroscope data
    """
    merged_data = []

    for _, gyro_row in gyro_df.iterrows():
            gyro_time = gyro_row['Time (s)']
            
            # Find the closest accelerometer reading in time
            time_diffs = np.abs(accel_df['Time (s)'] - gyro_time)
            closest_idx = time_diffs.idxmin()
            closest_time_diff = time_diffs.iloc[closest_idx]
            
            # Only merge if within tolerance
            if closest_time_diff <= time_tolerance:
                accel_row = accel_df.iloc[closest_idx]
                
                # Combine the data
                merged_row = {
                    'Time (s)': gyro_time,
                    'Time_Difference': closest_time_diff,
                    'X (rad/s)': gyro_row['X (rad/s)'],
                    'Y (rad/s)': gyro_row['Y (rad/s)'],
                    'Z (rad/s)': gyro_row['Z (rad/s)'],
                    'X (m/s^2)': accel_row['X (m/s^2)'],
                    'Y (m/s^2)': accel_row['Y (m/s^2)'],
                    'Z (m/s^2)': accel_row['Z (m/s^2)'],
                }
                
                # Add System Time if it exists
                if 'System Time' in accel_df.columns:
                    merged_row['System_Time'] = accel_row['System Time']
                    
                merged_data.append(merged_row)


    # Create merged DataFrame
    merged_df = pd.DataFrame(merged_data)

    # Print largest time difference if any
    # if not merged_df.empty:
    max_time_diff = merged_df['Time_Difference'].max()
        # if max_time_diff > time_tolerance:
    print(f"Warning: Maximum time difference in merged data is {max_time_diff} seconds.")
    
    return merged_df