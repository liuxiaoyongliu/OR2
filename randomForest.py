import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from gurobipy import Model, GRB

# ------------------------- global config ------------------------------------
b, h, M = 8, 2, 1e8
TREE_DEPTH = 9
N_TREES = 4
THRESHOLD = 0

alpha_list = np.arange(0.1, 1.0, 0.1)
v_list = np.arange(1500, 2501, 200)
top_k_list = range(1, 10)

# ------------------------- load sorted data ---------------------------------
df = pd.read_csv('real_dataset/X_Y_sorted_by_importance.csv')
X_all = df.drop(columns=['D', 'U'], errors='ignore').select_dtypes(include=[np.number]).values
y_all = df[['D', 'U']].values

# ----------------------------------------------------------------------------
def compute_weights(leaf_ids_target, leaf_ids_train, n_trees):
    n_train = leaf_ids_train.shape[0]
    weights = np.zeros(n_train)

    for col in range(0, leaf_ids_target.shape[0], 2):
        match = np.where(leaf_ids_train[:, col] == leaf_ids_target[col])[0]
        if match.size == 0:
            continue
        weights[match] += 1.0 / match.size

    weights /= n_trees
    return weights


def build_newsvendor_gurobi(d_vals, u_vals, weights, alpha, v_threshold):
    k_nn = len(d_vals)
    if k_nn == 0:
        return False, np.nan, np.nan

    model = Model()
    model.setParam('OutputFlag', 0)
    q = model.addVar(lb=0.0, name='q')
    v = model.addVars(k_nn, lb=0.0, name='v')
    o = model.addVars(k_nn, lb=0.0, name='o')
    gamma = model.addVars(k_nn, vtype=GRB.BINARY, name='gamma')

    for j in range(k_nn):
        model.addConstr(v[j] >= d_vals[j] - q * u_vals[j])
        model.addConstr(o[j] >= q * u_vals[j] - d_vals[j])
        model.addConstr(b * v[j] + h * o[j] + M * gamma[j] >= v_threshold)
        model.addConstr(b * v[j] + h * o[j] + M * gamma[j] <= M + v_threshold)

    model.addConstr(sum(float(weights[j]) * gamma[j] for j in range(k_nn)) >= alpha)
    model.setObjective(sum(float(weights[j]) * (b * v[j] + h * o[j]) for j in range(k_nn)), GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        q_star = q.X
        obj_val = model.ObjVal
        return True, q_star, obj_val
    return False, np.nan, np.nan

# ----------------------------------------------------------------------------
def evaluate_alpha_v_k(params):
    alpha, v_threshold, k_feat = params
    print(f"→ α={alpha:.1f}, v={v_threshold}, top_k={k_feat}", flush=True)

    # 只使用前 k_feat 个特征
    X = X_all[:3000, :k_feat]
    y = y_all[:3000, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(max_depth=TREE_DEPTH, n_estimators=N_TREES, random_state=42)
    reg.fit(X_train, y_train)
    leaf_ids_train = reg.apply(X_train)

    test_loss = 0.0
    test_obj = 0.0
    q_sum = 0.0
    cnt = 0

    for x_i, (d_i, u_i) in zip(X_test, y_test):
        leaf_ids_target = reg.apply(x_i.reshape(1, -1))[0]
        w_vec = compute_weights(leaf_ids_target, leaf_ids_train, N_TREES)
        idx = np.where(w_vec > THRESHOLD)[0]
        if idx.size == 0:
            continue

        d_vals, u_vals = y_train[idx].T
        weights = w_vec[idx]
        weights /= weights.sum()  # 建议归一化

        ok, q_star, obj_val = build_newsvendor_gurobi(d_vals, u_vals, weights, alpha, v_threshold)
        if ok:
            test_loss += b * max(d_i - q_star * u_i, 0) + h * max(q_star * u_i - d_i, 0)
            test_obj += obj_val
            q_sum += q_star
            cnt += 1

    if cnt > 0:
        return {
            'alpha': alpha,
            'v_threshold': v_threshold,
            'top_k_features': k_feat,
            'test_loss': test_loss / cnt,
            'test_obj_val': test_obj / cnt,
            'avg_q_pred': q_sum / cnt,
            'num_eval': cnt
        }
    else:
        return {
            'alpha': alpha,
            'v_threshold': v_threshold,
            'top_k_features': k_feat,
            'test_loss': np.nan,
            'test_obj_val': np.nan,
            'avg_q_pred': np.nan,
            'num_eval': 0
        }

# ----------------------------------------------------------------------------
def main():
    grid = list(product(alpha_list, v_list, top_k_list))
    results = []

    with ProcessPoolExecutor(max_workers=10) as exe:
        for res in exe.map(evaluate_alpha_v_k, grid):
            results.append(res)

    df_res = pd.DataFrame(results)
    df_res.to_csv('RandomForest/RF_best_feature_count_results.csv', index=False)
    print('\n✅ 并发计算完成，结果已保存 → RandomForest/RF_best_feature_count_results.csv')

if __name__ == '__main__':
    main()
