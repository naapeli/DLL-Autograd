import DLL

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target[:, np.newaxis], test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_dll = DLL.Tensor(X_train.tolist())
X_test_dll = DLL.Tensor(X_test.tolist())
y_train_dll = DLL.Tensor(y_train.tolist())
y_test_dll = DLL.Tensor(y_test.tolist())

hidden_size = 20
input_size = X_train.shape[1]
std = 0.1
n_steps = 30
n_epochs = 5

params = {
    "w1": DLL.randn((input_size, hidden_size), std=std),
    "b1": DLL.randn((hidden_size,), std=std),
    "w2": DLL.randn((hidden_size, hidden_size), std=std),
    "b2": DLL.randn((hidden_size,), std=std),
    "w3": DLL.randn((hidden_size, 1), std=std),
    "b3": DLL.randn((1,), std=std)
}

for param in params.values():
    param.requires_grad = True

def forward(X: DLL.Tensor):
    z1 = (X @ params["w1"] + params["b1"]).relu()
    z2 = (z1 @ params["w2"] + params["b2"]).relu()
    z3 = z2 @ params["w3"] + params["b3"]
    return z3.sigmoid()

lr = 1e-1
losses = []
accuracies = []

for epoch in range(n_epochs):
    for step in range(n_steps):
        length = y_train_dll.shape[0]
        y_train_batch = y_train_dll[int(step / n_steps * length): int(step / n_steps * length) + length // n_steps]
        X_train_batch = X_train_dll[int(step / n_steps * length): int(step / n_steps * length) + length // n_steps]
        y_pred = forward(X_train_batch)
        loss = ((y_train_batch - y_pred) ** 2).mean()
        loss.backward()
        
        for key in params:
            params[key] = params[key] - lr * params[key].grad
            params[key].zero_grad()
    
    test_pred_dll = forward(X_test_dll)
    test_loss = ((y_test_dll - test_pred_dll) ** 2).mean().item()
    
    test_pred_np = np.array(test_pred_dll.data).reshape(test_pred_dll.shape)
    y_test_np = np.array(y_test_dll.data).reshape(y_test_dll.shape)
    
    predictions = (test_pred_np > 0.5).astype(float)
    accuracy = (predictions == y_test_np).mean()
    
    losses.append(test_loss)
    accuracies.append(accuracy)

    print(f"Epoch {epoch + 1} | Loss: {test_loss:.4f} | Accuracy: {accuracy:.4f}")

fig, ax1 = plt.subplots()

ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss", color="tab:red")
ax1.plot(losses, color="tab:red", label="Test Loss")

ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy", color="tab:blue")
ax2.plot(accuracies, color="tab:blue", label="Test Accuracy")

plt.title("DLL Implementation Performance")
fig.tight_layout()
plt.show()
