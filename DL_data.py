import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.model_selection import KFold

label_map = {'walk': 0, 'bike': 1, 'metro': 2, 'lift': 3}
MAX_SEQ_LEN = 2048

# go through the cleaned_data and read files by category
def get_file_label_list(data_dir):
    file_label_list = []
    for label_name in os.listdir(data_dir):
        folder = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith('.csv'):
                file_label_list.append((os.path.join(folder, f), label_map[label_name]))
    return file_label_list

def get_kfolds(file_label_list, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    file_label_list = np.array(file_label_list, dtype=object)
    for train_idx, test_idx in kf.split(file_label_list):
        train_files = file_label_list[train_idx].tolist()
        test_files = file_label_list[test_idx].tolist()
        folds.append((train_files, test_files))
    return folds


# divide files into train/test in case to avoid data leakage
#def split_train_test(category_paths, n_test=1):
    #    train_files, test_files = [], []
    #    for label, files in category_paths.items():
    #        train = files[:-n_test]
    #        test = files[-n_test:]
    #        train_files.extend(train)
    #        test_files.extend(test)
    #    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")
#    return train_files, test_files


# keep the original time sequence
class SequenceDataset(Dataset):
    def __init__(self, file_label_list, scaler=None, fit_scaler=False):
        self.file_label_list = file_label_list
        self.scaler = scaler
        self.fit_scaler = fit_scaler

        if self.fit_scaler:
            all_data = []
            for fpath, _ in self.file_label_list:
                df = pd.read_csv(fpath)
                df = df.select_dtypes(include=['number'])
                all_data.append(df.values)
            concat_data = pd.concat([pd.DataFrame(arr) for arr in all_data])
            self.scaler = StandardScaler().fit(concat_data)

    def __len__(self):
        return len(self.file_label_list)

    def __getitem__(self, idx):
        fpath, label = self.file_label_list[idx]
        df = pd.read_csv(fpath)
        df = df.select_dtypes(include=['number'])
        x = df.values.astype('float32')
        if len(x) > MAX_SEQ_LEN:
            x = x[:MAX_SEQ_LEN]
        elif len(x) < MAX_SEQ_LEN:
            pad = np.zeros((MAX_SEQ_LEN - len(x), x.shape[1]), dtype=np.float32)
            x = np.vstack([x, pad])
        if self.scaler:
            x = self.scaler.transform(x)
        return torch.tensor(x), torch.tensor(label)


def create_dataloaders(train_files, test_files, batch_size=32, collate_fn=None):
    train_dataset = SequenceDataset(train_files, scaler=None, fit_scaler=True)
    test_dataset = SequenceDataset(test_files, scaler=train_dataset.scaler, fit_scaler=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn
    )
    return train_loader, test_loader