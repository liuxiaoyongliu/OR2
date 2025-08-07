import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, KFold
from gurobipy import Model, GRB
from concurrent.futures import ProcessPoolExecutor

# ==== 全局配置 ====
b, h, M = 8, 2, 1e8
K = 5
alpha_list = np.arange(0, 1.0, 0.1)
v_list = np.arange(1500, 2501, 200)
top_k_list = range(1, 10)  # 新增：前k个特征作为输入

OK_STAT = {
    GRB.OPTIMAL,
    GRB.SUBOPTIMAL,
    GRB.TIME_LIMIT,
    GRB.NODE_LIMIT,
    GRB.INTERRUPTED,
}
if hasattr(GRB, "SOLUTION_LIMIT"):
    OK_STAT.add(GRB.SOLUTION_LIMIT)

# ==== 数据准备 ====
df = pd.read_csv('real_dataset/X_Y_sorted_by_importance.csv')
X_all = df.drop(columns=['D', 'U'], errors='ignore').select_dtypes(include=[np.number]).values
y_all = df[['D', 'U']].values



# ==== 评估函数 ====
def evaluate_alpha_v_k(args):
    alpha, v_threshold, k_feat = args
    print(f"===alpha:{alpha}==========v:{v_threshold}=========k_feat:{k_feat}===============")
    X = X_all[:3000, :k_feat]  # 只取前k_feat个特征
    y = y_all[:3000, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    kf = KFold(n_splits=K, shuffle=False)
    k_nn = 50
    k = 50
    min_cv_loss = float('inf')
    best_sigma = None

    for sigma in np.arange(0.1, 3.1, 0.1):
        fold_losses = []
        for train_idx, val_idx in kf.split(X_train):
            ref_X, ref_y = X_train[train_idx], y_train[train_idx]
            val_X, val_y = X_train[val_idx], y_train[val_idx]
            fold_loss = 0
            cnt = 0
            for i, x_i in enumerate(val_X):
                d_i, u_i = val_y[i]
                dists = np.linalg.norm(ref_X - x_i, axis=1)
                ke = np.exp(-dists ** 2 / (2 * sigma ** 2))
                nearest_idx = np.argsort(-ke)[:k]
                neighbor_y = ref_y[nearest_idx]
                neighbor_dist = dists[nearest_idx]
                weights = np.exp(-neighbor_dist ** 2 / (2 * sigma ** 2))
                d_vals = neighbor_y[:, 0]
                u_vals = neighbor_y[:, 1]

                model = Model()
                model.setParam("OutputFlag", 0)
                model.setParam('MIPGap', 0.05)
                model.setParam('PoolSearchMode', 2)
                model.setParam('PoolSolutions', 10)
                q = model.addVar(lb=0.0, name="q")
                v = model.addVars(k_nn, lb=0.0)
                o = model.addVars(k_nn, lb=0.0)
                gamma = model.addVars(k_nn, vtype=GRB.BINARY)

                for j in range(k_nn):
                    model.addConstr(v[j] >= d_vals[j] - q * u_vals[j])
                    model.addConstr(o[j] >= q * u_vals[j] - d_vals[j])
                    model.addConstr(b * v[j] + h * o[j] + M * gamma[j] >= v_threshold)
                    model.addConstr(b * v[j] + h * o[j] + M * gamma[j] <= M + v_threshold)

                model.addConstr(sum(weights[j] * gamma[j] for j in range(k_nn)) >= alpha)
                model.setObjective(sum(weights[j] * (b * v[j] + h * o[j]) for j in range(k_nn)), GRB.MINIMIZE)

                try:
                    model.optimize()
                except:
                    continue

                if model.status in OK_STAT and model.SolCount > 0:
                    q_pred = q.X
                    loss = b * max(d_i - q_pred * u_i, 0) + h * max(q_pred * u_i - d_i, 0)
                    fold_loss += loss
                    cnt += 1

            if cnt > 0:
                fold_losses.append(fold_loss / cnt)

        mean_loss = np.mean(fold_losses)
        if mean_loss < min_cv_loss:
            min_cv_loss = mean_loss
            best_sigma = sigma

    # ==== 测试集评估 ====
    test_loss, obj_val, q_pred_val, cnt = 0, 0, 0, 0
    for i, x_i in enumerate(X_test):
        d_i, u_i = y_test[i]
        dists = np.linalg.norm(X_train - x_i, axis=1)
        ke = np.exp(-dists ** 2 / (2 * best_sigma ** 2))
        nearest_idx = np.argsort(-ke)[:k]
        neighbor_y = y_train[nearest_idx]
        neighbor_dist = dists[nearest_idx]
        weights = np.exp(-neighbor_dist ** 2 / (2 * best_sigma ** 2))
        d_vals = neighbor_y[:, 0]
        u_vals = neighbor_y[:, 1]

        model = Model()
        model.setParam("OutputFlag", 0)
        model.setParam('MIPGap', 0.05)
        model.setParam('PoolSearchMode', 2)
        model.setParam('PoolSolutions', 10)
        q = model.addVar(lb=0.0, name="q")
        v = model.addVars(k_nn, lb=0.0)
        o = model.addVars(k_nn, lb=0.0)
        gamma = model.addVars(k_nn, vtype=GRB.BINARY)

        for j in range(k_nn):
            model.addConstr(v[j] >= d_vals[j] - q * u_vals[j])
            model.addConstr(o[j] >= q * u_vals[j] - d_vals[j])
            model.addConstr(b * v[j] + h * o[j] + M * gamma[j] >= v_threshold)
            model.addConstr(b * v[j] + h * o[j] + M * gamma[j] <= M + v_threshold)

        model.addConstr(sum(weights[j] * gamma[j] for j in range(k_nn)) >= alpha)
        model.setObjective(sum(weights[j] * (b * v[j] + h * o[j]) for j in range(k_nn)), GRB.MINIMIZE)

        try:
            model.optimize()
        except:
            continue

        if model.status in OK_STAT and model.SolCount > 0:
            q_pred = q.X
            test_loss += b * max(d_i - q_pred * u_i, 0) + h * max(q_pred * u_i - d_i, 0)
            obj_val += sum(weights[j] * (b * v[j].X + h * o[j].X) for j in range(k_nn))
            q_pred_val += q_pred
            cnt += 1

    return {
        "alpha": alpha,
        "v_threshold": v_threshold,
        "top_k_features": k_feat,
        "best_sigma": best_sigma,
        "best_train_mean_loss": min_cv_loss,
        "best_test_loss": test_loss / cnt if cnt else np.nan,
        "avg_obj_val": obj_val / cnt if cnt else np.nan,
        "avg_q_pred": q_pred_val / cnt if cnt else np.nan
    }

# ==== 主程序入口 ====
if __name__ == '__main__':
    param_grid = list(product(alpha_list, v_list, top_k_list))
    results = []

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = list(executor.map(evaluate_alpha_v_k, param_grid))

    for res in futures:
        if res:
            results.append(res)

    pd.DataFrame(results).to_csv("KR/KR_best_feature_count_results.csv", index=False)
    print("并发运行完成，结果已保存。")