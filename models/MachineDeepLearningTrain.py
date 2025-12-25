import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class SDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X, dtype=np.float32))
        self.y = torch.tensor(np.array(y, dtype=np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def deepLearningTrain(model,  X, Y, X_test, Y_test, isVDCNN=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = SDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    model_path = 'models/ckpt/model.pth'

    num_epochs = 300
    patience = 40
    best_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0
        for inputs, targets in dataloader:
            if isVDCNN:
                inputs = inputs.int().clamp(min=1, max=99)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算正确率
            total_correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += targets.size(0)
        accuracy = total_correct / total_samples
        print(f"{epoch + 1}/{num_epochs}，Accuracy: {accuracy:.4f}")

        # 检查是否早停
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)  # 保存模型参数
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    print("=================test=================")

    dataset_test = SDataset(X_test, Y_test)
    dataloader = DataLoader(dataset_test, batch_size=1024, shuffle=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式
    total_correct = 0
    total_samples = 0
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            if isVDCNN:
                inputs = inputs.int().clamp(min=1, max=99)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            total_correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += targets.size(0)

            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")

    df = pd.DataFrame({
        'predict': all_predictions,
        'label': Y_test
    })

    df.to_excel('static/labels_predict.xlsx', index=False)

    return accuracy


def machineLearningTrain(model,  X, Y, X_test, Y_test, isCRF=False):
    if isCRF:
        X_list = X.tolist()
        Y_list = Y.tolist()
        # 为每个标记创建特征字典
        X_features = []
        for sequence in X_list:
            X_features.append([{'feature' + str(i): value for i, value in enumerate(sequence)}])
        Y_labels = [str(label) for label in Y_list]
        model.fit(X_features, Y_labels)

        X_test_list = X_test.tolist()
        X_test_features = []
        for sequence in X_test_list:
            X_test_features.append([{'feature' + str(i): value for i, value in enumerate(sequence)}])
        X_test = X_test_features

    if isCRF is False:
        model.fit(X, Y)

    Y_pred = model.predict(X_test)

    if isinstance(Y_pred, list):
        Y_pred = np.array(Y_pred, dtype=int).ravel()
    accuracy = accuracy_score(Y_test, Y_pred)
    all_predictions = Y_pred

    df = pd.DataFrame({
        'predict': all_predictions,
        'label': Y_test
    })

    df.to_excel('static/labels_predict.xlsx', index=False)

    return accuracy
