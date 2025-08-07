import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import NearestNeighbors
from gurobipy import Model, GRB
from concurrent.futures import ProcessPoolExecutor
# ==== 配置 ====
b, h, M = 8, 2, 1e8
sigma = 0.5
K = 5
weighted = False
alpha_list = np.arange(0, 1.0, 0.1)
v_list = np.arange(1500, 2501, 200)
top_k_list = range(1, 11)
K_MAX = 40

OK_STAT = {
    GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT,
    GRB.NODE_LIMIT, GRB.INTERRUPTED
}
if hasattr(GRB, "SOLUTION_LIMIT"):
    OK_STAT.add(GRB.SOLUTION_LIMIT)

# ==== 数据准备 ====
df = pd.read_csv('real_dataset/X_Y_sorted_by_importance.csv')
X_full = df.drop(columns=['D', 'U'], errors='ignore').select_dtypes(include=[np.number]).values
y = df[['D', 'U']].values
# 只使用一半数据
X_full = X_full[:3000, :]  # 只取前k_feat个特征
y = y[:3000, :]


# 再进行训练/测试划分
X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

def precompute_neighbors(X_train, X_test, k_max=K_MAX):
    nbrs_train = NearestNeighbors(n_neighbors=k_max, algorithm='auto').fit(X_train)
    train_dists, train_indices = nbrs_train.kneighbors(X_train)
    test_dists, test_indices = nbrs_train.kneighbors(X_test)
    return train_dists, train_indices, test_dists, test_indices
# ==== 核心函数 ====
def build_model(k_nn, weights, d_vals, u_vals, alpha, v_threshold):
    model = Model()
    model.setParam("OutputFlag", 0)
    model.setParam('MIPGap', 0.05)
    model.setParam('PoolSearchMode', 2)
    model.setParam('PoolSolutions', 10)

    q = model.addVar(lb=0.0, name="q")
    v = model.addVars(k_nn, lb=0.0, name="v")
    o = model.addVars(k_nn, lb=0.0, name="o")
    gamma = model.addVars(k_nn, vtype=GRB.BINARY, name="gamma")

    for j in range(k_nn):
        model.addConstr(v[j] >= d_vals[j] - q * u_vals[j])
        model.addConstr(o[j] >= q * u_vals[j] - d_vals[j])
        model.addConstr(b * v[j] + h * o[j] + M * gamma[j] >= v_threshold)
        model.addConstr(b * v[j] + h * o[j] + M * gamma[j] <= M + v_threshold)

    model.addConstr(sum(weights[j] * gamma[j] for j in range(k_nn)) >= alpha)
    model.setObjective(sum(weights[j] * (b * v[j] + h * o[j]) for j in range(k_nn)), GRB.MINIMIZE)

    return model, q, v, o
def evaluate_alpha_v(args):
    alpha, v_threshold, top_k_features = args

    X_train = X_train_full[:, :top_k_features]
    X_test = X_test_full[:, :top_k_features]

    train_dists, train_indices, test_dists, test_indices = precompute_neighbors(X_train, X_test)

    kf = KFold(n_splits=K, shuffle=False)
    k_loss_dict = {}
    best_k = None
    min_cv_loss = float('inf')

    for k in range(5, K_MAX + 1):
        print(f"[alpha={alpha:.2f} | v={v_threshold} | top_k={top_k_features}] >> Evaluating k={k}")
        fold_losses = []

        for train_idx, val_idx in kf.split(X_train):
            ref_X, ref_y = X_train[train_idx], y_train[train_idx]
            val_X, val_y = X_train[val_idx], y_train[val_idx]

            fold_loss = 0
            cnt = 0

            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(ref_X)
            val_dists, val_indices = nbrs.kneighbors(val_X)

            for i, x_i in enumerate(val_X):
                d_i, u_i = val_y[i]
                idxs = val_indices[i]
                dists = val_dists[i]

                neighbor_y = ref_y[idxs]
                d_vals = neighbor_y[:, 0]
                u_vals = neighbor_y[:, 1]
                weights = np.exp(-dists ** 2 / (2 * sigma ** 2)) if weighted else np.ones_like(d_vals) / k

                try:
                    model, q, v, o = build_model(k, weights, d_vals, u_vals, alpha, v_threshold)
                    model.optimize()
                except:
                    continue

                if model.status in OK_STAT and model.SolCount > 0:
                    q_pred = q.X
                    loss = b * max(d_i - q_pred * u_i, 0) + h * max(q_pred * u_i - d_i, 0)
                    fold_loss += loss
                    cnt += 1

            if cnt:
                fold_losses.append(fold_loss / cnt)

        mean_loss = np.mean(fold_losses)
        k_loss_dict[k] = mean_loss
        if mean_loss < min_cv_loss:
            min_cv_loss = mean_loss
            best_k = k

    # ==== 测试集评估 ====
    test_loss = obj_val = q_pred_val = cnt = 0
    for i, x_i in enumerate(X_test):
        d_i, u_i = y_test[i]
        idxs = test_indices[i][:best_k]
        dists = test_dists[i][:best_k]

        neighbor_y = y_train[idxs]
        d_vals = neighbor_y[:, 0]
        u_vals = neighbor_y[:, 1]
        weights = np.exp(-dists ** 2 / (2 * sigma ** 2)) if weighted else np.ones_like(d_vals) / best_k

        try:
            model, q, v, o = build_model(best_k, weights, d_vals, u_vals, alpha, v_threshold)
            model.optimize()
        except:
            continue

        if model.status in OK_STAT and model.SolCount > 0:
            q_pred = q.X
            loss = b * max(d_i - q_pred * u_i, 0) + h * max(q_pred * u_i - d_i, 0)
            test_loss += loss
            obj_val += sum(weights[j] * (b * v[j].X + h * o[j].X) for j in range(best_k))
            q_pred_val += q_pred
            cnt += 1

    return {
        "alpha": alpha,
        "v_threshold": v_threshold,
        "top_k_features": top_k_features,
        "best_k": best_k,
        "best_train_mean_loss": min_cv_loss,
        "best_test_loss": test_loss / cnt if cnt else np.nan,
        "avg_obj_val": obj_val / cnt if cnt else np.nan,
        "avg_q_pred": q_pred_val / cnt if cnt else np.nan
    }


# ==== 主逻辑 ====
if __name__ == '__main__':
    param_grid = list(product(alpha_list, v_list, top_k_list))
    results = []

    with ProcessPoolExecutor(max_workers=20) as executor:
        for res in executor.map(evaluate_alpha_v, param_grid):
            if res:
                results.append(res)

    pd.DataFrame(results).to_csv("KNN/best_k_with_feature_selection.csv", index=False)
    print("全部完成，结果保存为 best_k_with_feature_selection.csv")
