import pandas as pd
import numpy as np

def calculate_sampling_frequency(csv_file_path):
    """
    Calculate the sampling frequency from accelerometer data CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
    
    Returns:
        dict: Dictionary containing sampling frequency and related statistics
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get the time column (assuming it's the first column)
    time_column = df.iloc[:, 0]  # First column
    
    # Calculate time differences between consecutive samples
    time_diffs = np.diff(time_column)
    
    # Calculate average time interval
    avg_time_interval = np.mean(time_diffs)
    
    # Calculate sampling frequency (Hz)
    sampling_frequency = 1.0 / avg_time_interval
    
    # Calculate some additional statistics
    min_interval = np.min(time_diffs)
    max_interval = np.max(time_diffs)
    std_interval = np.std(time_diffs)
    
    # Check if sampling is relatively uniform
    coefficient_of_variation = std_interval / avg_time_interval
    is_uniform = coefficient_of_variation < 0.01  # Less than 1% variation
    
    results = {
        'sampling_frequency_hz': sampling_frequency,
        'average_time_interval_s': avg_time_interval,
        'min_time_interval_s': min_interval,
        'max_time_interval_s': max_interval,
        'std_time_interval_s': std_interval,
        'coefficient_of_variation': coefficient_of_variation,
        'is_sampling_uniform': is_uniform,
        'total_samples': len(time_column),
        'total_duration_s': time_column.iloc[-1] - time_column.iloc[0]
    }
    
    return results

def print_results(results):
    """Print the sampling frequency results in a readable format."""
    print("=== Sampling Frequency Analysis ===")
    print(f"Sampling Frequency: {results['sampling_frequency_hz']:.2f} Hz")
    print(f"Average Time Interval: {results['average_time_interval_s']:.6f} seconds")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Total Duration: {results['total_duration_s']:.3f} seconds")
    print()
    print("=== Timing Statistics ===")
    print(f"Min Time Interval: {results['min_time_interval_s']:.6f} seconds")
    print(f"Max Time Interval: {results['max_time_interval_s']:.6f} seconds")
    print(f"Standard Deviation: {results['std_time_interval_s']:.6f} seconds")
    print(f"Coefficient of Variation: {results['coefficient_of_variation']:.4f}")
    print(f"Uniform Sampling: {'Yes' if results['is_sampling_uniform'] else 'No'}")

# Example usage
if __name__ == "__main__":
    # Replace 'your_file.csv' with the path to your CSV file
    csv_file_path = 'data/Gyroscope.csv'
    csv_file_path1 = 'data/Linear Acceleration.csv'
    
    try:
        results = calculate_sampling_frequency(csv_file_path)
        print_results(results)
        
        # You can also access individual values
        print(f"\nQuick result: {results['sampling_frequency_hz']:.1f} Hz")


        results1 = calculate_sampling_frequency(csv_file_path1)
        print_results(results1)

        # You can also access individual values
        print(f"\nQuick result: {results1['sampling_frequency_hz']:.1f} Hz")

    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file_path}'")
        print("Please make sure the file path is correct.")
    except Exception as e:
        print(f"Error processing file: {e}")
