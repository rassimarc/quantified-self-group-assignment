import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# by file names
walk_files = glob.glob('feature_output/walk/*.csv')
bike_files = glob.glob('feature_output/bike/*.csv')
lift_files = glob.glob('feature_output/lift/*.csv')
metro_files = glob.glob('feature_output/metro/*.csv')
all_files = walk_files + bike_files + lift_files + metro_files

def split_files(files, n_test=1, random_state=42):
    files = list(files)
    files = shuffle(files, random_state=random_state)
    return files[:-n_test], files[-n_test:]  # train_files, test_files

# split train/test set in files level by category.
walk_train, walk_test = split_files(walk_files, n_test=1)
bike_train, bike_test = split_files(bike_files, n_test=1)
lift_train, lift_test = split_files(lift_files, n_test=1)
metro_train, metro_test = split_files(metro_files, n_test=1)

train_files = walk_train + bike_train + lift_train + metro_train
test_files = walk_test + bike_test + lift_test + metro_test

print(f'Train files: {len(train_files)}, Test files: {len(test_files)}')

# load and append
def load_files(file_list):
    dfs = []
    for f in file_list:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

train_df = load_files(train_files)
test_df = load_files(test_files)

# features and labels
X_train = train_df.drop(columns=['label'])
y_train = train_df['label']
X_test = test_df.drop(columns=['label'])
y_test = test_df['label']

# standard, fit in train, transform in test.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('Train size:', X_train.shape, 'Test size:', X_test.shape)

# decision tree
print("\n=== Decision Tree ===")
param_grid_tree = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10]
}
dt = DecisionTreeClassifier(random_state=42)
grid_tree = GridSearchCV(dt, param_grid_tree, cv=5, scoring='f1_macro')
grid_tree.fit(X_train, y_train)
y_pred_tree = grid_tree.predict(X_test)
print(classification_report(y_test, y_pred_tree))

#KNN
print("\n=== K-Nearest Neighbors ===")
param_grid_knn = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"]
}
knn = KNeighborsClassifier()
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='f1_macro')
grid_knn.fit(X_train_scaled, y_train)
y_pred_knn = grid_knn.predict(X_test_scaled)
print(classification_report(y_test, y_pred_knn))

#RF
print("\n=== Random Forest ===")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf, zero_division=0))


#NB

print("\n=== Naive Bayes ===")
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print(classification_report(y_test, y_pred_nb, zero_division=0))


# evaluation
def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "F1-score": f1_score(y_true, y_pred, average="macro")
    }

results = [
    evaluate_model("Decision Tree", y_test, y_pred_tree),
    evaluate_model("KNN", y_test, y_pred_knn),
    evaluate_model("RF", y_test, y_pred_rf),
    evaluate_model("NB", y_test, y_pred_nb)
]

df_results = pd.DataFrame(results)
print("\n=== Summary ===")
print(df_results)

print("Best Decision Tree parameters:", grid_tree.best_params_)
print("Best KNN parameters:", grid_knn.best_params_)
print("Best Random Forest parameters: used default (n_estimators=100)")
print("Naive Bayes: no hyperparameters tuned.")
