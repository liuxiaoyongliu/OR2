import pandas as pd
import numpy as np

# 读取数据（替换为你的路径）
df = pd.read_csv("data.csv")

# 1. 计算供货稳定概率
timely = df['on_time_delivery_rate'].clip(0, 1)
quality = (1 - df['defect_rate']).clip(0, 1)
capacity = np.minimum(df['items_offered'] / df['items_requested'].replace(0, np.nan), 1).fillna(0)
lead_var = df['lead_time_variance'].fillna(0)
lead_factor = 1 / (1 + lead_var)

df['u'] = timely * quality * capacity * lead_factor
df['D'] = df['items_requested']
# 2. 构造二维目标变量 Y
Y = df[['D', 'u']].to_numpy()
# === 2. 构造输入特征 X（去掉目标变量 D 和 u）===
drop_cols = ['items_requested', 'u', 'D']
X = df.drop(columns=drop_cols, errors='ignore')

# 可选：转为 numpy
X = X.select_dtypes(include=[np.number]).to_numpy()  # 仅使用数值型特征（可选）

X_df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
Y_df = pd.DataFrame(Y, columns=['D', 'U'])  # 二维目标变量列名

# 2. 合并 X 和 Y
combined_df = pd.concat([X_df, Y_df], axis=1)

# 3. 保存为 CSV
combined_df.to_csv("X_Y_combined.csv", index=False)
# 可选：打印前几行
# print("Y 变量（前5行）：")
# print(Y[:5])
