import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader,
                num_epochs, lr, device='cuda'):
    """
    Train a CNN+LSTM model for emotion recognition with progress bars.

    Args:
        model: PyTorch nn.Module
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        num_epochs: number of epochs
        lr: learning rate
        device: 'cuda' or 'cpu'
    """

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop with tqdm
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
        for inputs, labels in train_loader_iter:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            train_loader_iter.set_postfix({'loss': running_loss / total, 'acc': correct / total})

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation loop with tqdm
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_loader_iter = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for val_inputs, val_labels in val_loader_iter:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_labels)

                val_loss += v_loss.item() * val_inputs.size(0)
                _, val_pred = val_outputs.max(1)
                val_correct += (val_pred == val_labels).sum().item()
                val_total += val_labels.size(0)

                val_loader_iter.set_postfix({'val_loss': val_loss / val_total, 'val_acc': val_correct / val_total})

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Plot loss and accuracy curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" Using device ", device)

param_grid = {
    'lr': [1e-2, 1e-3, 1e-4],
    'hidden_size': [64, 128, 256],
    'num_layers': [2, 3, 4],
    'in_channels': [1],
    'out_channels': [2],
    'kernel_size': [3, 5],
    'num_epochs': [10]
}

for lr in param_grid['lr']:
    for hidden_layer_size in param_grid['hidden_size']:
        for num_conv_layers in param_grid['num_layers']:
            for in_channels in param_grid['in_channels']:
                for out_channels in param_grid['out_channels']:
                    for kernel_size in param_grid['kernel_size']:
                        for num_epochs in param_grid['num_epochs']:
                            m = EmotionDetector(num_conv_layers, in_channels, out_channels, kernel_size).to(device)
                            train_model(m, train_loader, val_loader,
                                        num_epochs=10, lr, device=device)