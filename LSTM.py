import torch
import torch.nn as nn

from DL_data import get_file_label_list, get_kfolds, create_dataloaders
from sklearn.model_selection import StratifiedKFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = './cleaned_data'

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_correct += (y_hat.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, total_correct / total

def evaluate_model(model, test_loader):
    model.eval()
    total_correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            total_correct += (y_hat.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_correct / total

def main():
    file_label_list = get_file_label_list(data_dir)
    files = [f for f, _ in file_label_list]
    labels = [l for _, l in file_label_list]

    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    fold_accs = []
    for fold_idx, (train_index, test_index) in enumerate(kf.split(files, labels), start=1):
        print(f"\n=== Fold {fold_idx} ===")

        train_files = [files[i] for i in train_index]
        train_labels = [labels[i] for i in train_index]
        test_files = [files[i] for i in test_index]
        test_labels = [labels[i] for i in test_index]

        train_file_label_list = list(zip(train_files, train_labels))
        test_file_label_list = list(zip(test_files, test_labels))

        train_loader, test_loader = create_dataloaders(train_file_label_list, test_file_label_list, batch_size=32)

        example_batch, _ = next(iter(train_loader))
        input_size = example_batch.shape[2]

        model = LSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=4,
            num_classes=4
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(10):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            test_acc = evaluate_model(model, test_loader)
            print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

        fold_accs.append(test_acc)

    avg_acc = sum(fold_accs) / len(fold_accs)
    print(f'\nAverage Test Accuracy across {len(fold_accs)} folds: {avg_acc:.4f}')


if __name__ == "__main__":
    main()
