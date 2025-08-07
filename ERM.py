import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ProcessPoolExecutor

# ==== 全局参数 ====
b, h = 8, 2
top_k_list = range(1, 11)
N_USE = 1000
EPOCHS = 10000
LR = 0.05

# ==== 自定义损失函数（newsvendor）====
def custom_loss(y_pred, y_true):
    y_true2 = torch.tensor(y_true, dtype=torch.float)
    d = y_true2[:, 0]
    u = y_true2[:, 1]
    return torch.mean(
        b * torch.clamp(d - y_pred.squeeze() * u, min=0) +
        h * torch.clamp(y_pred.squeeze() * u - d, min=0)
    )

# ==== 读取数据（与 KR 一致）====
df = pd.read_csv('real_dataset/X_Y_sorted_by_importance.csv')
X_all = df.drop(columns=['D', 'U'], errors='ignore').select_dtypes(include=[np.number]).values
y_all = df[['D', 'U']].values

# ==== 单模型评估逻辑（可并行）====
def evaluate_top_k(k_feat):
    print(f"训练中：top_k_features = {k_feat}")
    X = X_all[:N_USE, :k_feat]
    y = y_all[:N_USE, :]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = nn.Linear(k_feat, 1)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        y_pred = model(torch.tensor(X_train, dtype=torch.float))
        loss = custom_loss(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 2000 == 0:
            print(f"[k={k_feat}] Epoch {epoch} Loss: {loss.item():.4f}")

    # 测试集评估
    with torch.no_grad():
        y_pred_test = model(torch.tensor(X_test, dtype=torch.float))
        loss = custom_loss(y_pred_test, y_test).item()
        avg_q = torch.mean(y_pred_test).item()

    return {
        "top_k_features": k_feat,
        "test_loss": loss,
        "avg_q_pred": avg_q
    }

# ==== 并行运行 ====
if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(evaluate_top_k, top_k_list))

    pd.DataFrame(results).to_csv("ERM/ERM_top_k_feature_results.csv", index=False)
    print("所有特征数遍历完毕，结果已保存。")
