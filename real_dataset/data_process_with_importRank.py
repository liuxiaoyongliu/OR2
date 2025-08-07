import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
#####
#  1.存储的 csv 文件, 如果是数值变量 就归一化去除量纲， 如果是 分类变量 就 one-hot 编码
#  2. 通过 训练一个随机森林，来对不同的特征进行重要性排序， 按照顺序存入 X_Y_sorted_by_importance 文件中
#####

# 1. 读取数据
df = pd.read_csv("X_Y_combined.csv")

# 2. 拆分输入和目标
X_raw = df.drop(columns=['D', 'U'], errors='ignore')
Y = df[['D', 'U']].values

# 3. 自动识别类别型与数值型特征
categorical_cols = []
numerical_cols = []
for col in X_raw.columns:
    unique_vals = X_raw[col].dropna().unique()
    if len(unique_vals) <= 10 and np.all(np.equal(np.mod(unique_vals, 1), 0)):
        categorical_cols.append(col)
    else:
        numerical_cols.append(col)

# 4. 创建预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ]
)

# 5. 对 X 进行预处理（数值归一化 + One-Hot）
X_processed = preprocessor.fit_transform(X_raw)

# 6. 获取完整的特征名
feature_names_num = numerical_cols
feature_names_cat = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
all_feature_names = feature_names_num + feature_names_cat

# 7. 训练随机森林并提取特征重要性（目标变量：D）
rf = RandomForestRegressor(random_state=42)
rf.fit(X_processed, Y[:, 0])
importances = rf.feature_importances_

# 8. 对特征按重要性排序
sorted_indices = np.argsort(importances)[::-1]
sorted_feature_names = [all_feature_names[i] for i in sorted_indices]

# 9. 构造新的 DataFrame（按重要性排序的特征 + D + U）
X_sorted = X_processed[:, sorted_indices]
X_sorted_df = pd.DataFrame(X_sorted, columns=sorted_feature_names)
Y_df = pd.DataFrame(Y, columns=['D', 'U'])
combined_sorted_df = pd.concat([X_sorted_df, Y_df], axis=1)

# 10. 保存为新的 CSV 文件
combined_sorted_df.to_csv("X_Y_sorted_by_importance.csv", index=False)
print("已保存为：X_Y_sorted_by_importance.csv")
