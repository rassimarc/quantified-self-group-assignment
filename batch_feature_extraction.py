
import os
import pandas as pd
from all_feature_extractors import extract_combined_features

def process_all_data(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    for label in os.listdir(input_root):
        label_folder = os.path.join(input_root, label)
        if not os.path.isdir(label_folder):
            continue

        for file in os.listdir(label_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(label_folder, file)
                try:
                    df = pd.read_csv(file_path)
                    features = extract_combined_features(df)

                    features['label'] = label  # add label
                    output_file = os.path.join(output_root, f'{label}_{file}')
                    features.to_csv(output_file, index=False)
                    print(f" Processed: {file} → {output_file}")
                except Exception as e:
                    print(f" Error processing {file_path}: {e}")

if __name__ == "__main__":
    input_dir = "/Users/ming/Projects/mlqs/clean_data"  # input
    output_dir = "/Users/ming/Projects/mlqs/feature_output"  # output
    process_all_data(input_dir, output_dir)
