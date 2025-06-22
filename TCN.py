import torch
import torch.nn as nn
from DL_data import get_file_label_list, get_kfolds, create_dataloaders
from sklearn.model_selection import StratifiedKFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = './cleaned_data'

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, padding, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))
        if out.size(2) != res.size(2):
            diff = res.size(2) - out.size(2)
            if diff > 0:
                res = res[:, :, :out.size(2)]
            else:
                out = out[:, :, :res.size(2)]
        return out + res

class TCNModel(nn.Module):
    def __init__(self, input_size, num_classes, num_channels=64, num_levels=4, kernel_size=4, dropout=0.3):
        super(TCNModel, self).__init__()

        layers = []
        in_channels = input_size
        for i in range(num_levels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_channels, num_channels, kernel_size, dilation, padding, dropout))
            in_channels = num_channels

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        # x: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out.mean(dim=2)
        return self.fc(out)

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

        model = TCNModel(
            input_size=input_size,
            num_classes=4,
            num_channels=64,
            num_levels=4,
            kernel_size=4,
            dropout=0.3
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

if __name__ == '__main__':
    main()
