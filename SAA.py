import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from gurobipy import Model, GRB
from concurrent.futures import ProcessPoolExecutor

# ==== 全局配置 ====
b, h, M = 8, 2, 1e8
alpha_list = np.arange(0, 1.0, 0.1)
v_list = np.arange(1500, 2501, 200)
top_k_list = range(1, 10)  # 遍历前 k 个特征
N_USE = 3000               # 与 KR 一致：只取前 1000 条

OK_STAT = {
    GRB.OPTIMAL,
    GRB.SUBOPTIMAL,
    GRB.TIME_LIMIT,
    GRB.NODE_LIMIT,
    GRB.INTERRUPTED,
}
if hasattr(GRB, "SOLUTION_LIMIT"):
    OK_STAT.add(GRB.SOLUTION_LIMIT)

# ==== 数据准备（与 KR 一致）====
# 文件：real_dataset/X_Y_sorted_by_importance.csv
# X：去掉 D、U 的所有数值列（按重要性已排序）
# y：两列目标 ['D','U']
df = pd.read_csv('real_dataset/X_Y_sorted_by_importance.csv')
X_all = df.drop(columns=['D', 'U'], errors='ignore').select_dtypes(include=[np.number]).values
y_all = df[['D', 'U']].values

def evaluate_alpha_v_k(args):
    alpha, v_threshold, k_feat = args
    print(f"=== alpha:{alpha}  v:{v_threshold}  top_k:{k_feat} ===")

    # 与 KR 一致：截取前 N_USE 条、前 k_feat 个特征
    X = X_all[:N_USE, :k_feat]
    y = y_all[:N_USE, :]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # SAA：对每个测试样本，从训练集中随机抽样 k_nn 个“情景”（邻居）
    k_nn = min(500, len(X_train))  # 保护：样本不足时取最小可用
    test_loss = 0.0
    obj_val = 0.0
    q_pred_val = 0.0
    cnt = 0

    for i, x_i in enumerate(X_test):
        d_i, u_i = y_test[i]

        # 随机抽样 k_nn 个训练索引（不放回）
        rand_indices = np.random.choice(len(X_train), size=k_nn, replace=False)
        neighbor_y = y_train[rand_indices]
        d_vals = neighbor_y[:, 0]
        u_vals = neighbor_y[:, 1]

        model = Model()
        model.setParam("OutputFlag", 0)
        model.setParam('MIPGap', 0.05)

        q = model.addVar(lb=0.0, name="q")
        v = model.addVars(k_nn, lb=0.0)
        o = model.addVars(k_nn, lb=0.0)
        gamma = model.addVars(k_nn, vtype=GRB.BINARY)

        for j in range(k_nn):
            model.addConstr(v[j] >= d_vals[j] - q * u_vals[j])
            model.addConstr(o[j] >= q * u_vals[j] - d_vals[j])
            model.addConstr(b * v[j] + h * o[j] + M * gamma[j] >= v_threshold)
            model.addConstr(b * v[j] + h * o[j] + M * gamma[j] <= M + v_threshold)

        # SAA：计数约束（不加权）
        model.addConstr(sum(gamma[j] for j in range(k_nn)) >= alpha)
        model.setObjective(sum((b * v[j] + h * o[j]) for j in range(k_nn)), GRB.MINIMIZE)

        try:
            model.optimize()
        except Exception:
            continue

        if model.status in OK_STAT and model.SolCount > 0:
            q_pred = q.X
            # 评估损失（真实 D,U 下的 newsvendor 损失）
            loss = b * max(d_i - q_pred * u_i, 0) + h * max(q_pred * u_i - d_i, 0)
            test_loss += loss
            # 记录目标值与 q
            obj_val += sum((b * v[j].X + h * o[j].X) for j in range(k_nn))
            q_pred_val += q_pred
            cnt += 1

    return {
        "alpha": alpha,
        "v_threshold": v_threshold,
        "top_k_features": k_feat,
        "best_test_loss": (test_loss / cnt) if cnt else np.nan,
        "avg_obj_val": (obj_val / cnt) if cnt else np.nan,
        "avg_q_pred": (q_pred_val / cnt) if cnt else np.nan,
        "num_eval": cnt,
    }

# ==== 主执行逻辑 ====
if __name__ == '__main__':
    param_grid = list(product(alpha_list, v_list, top_k_list))
    results = []

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = list(executor.map(evaluate_alpha_v_k, param_grid))

    for res in futures:
        if res:
            results.append(res)

    pd.DataFrame(results).to_csv("SAA/SAA_best_feature_count_results.csv", index=False)
    print("并发计算完成，结果已保存。")
