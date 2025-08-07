import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, KFold
from gurobipy import Model, GRB
from concurrent.futures import ProcessPoolExecutor
from sklearn.tree import DecisionTreeRegressor
###
# 核方法 / KNN
###

# ==== 全局配置 ====
b, h, M = 8, 2, 1e8
K = 5
deep = 10 # 树高度
alpha_list = np.arange(0, 1.0, 0.1)
v_list = np.arange(1500, 2501, 200)
top_k_list = range(1, 10)
# ==== 数据准备 ====
df = pd.read_csv('real_dataset/X_Y_sorted_by_importance.csv')  # 用排序后的特征
X_all = df.drop(columns=['D', 'U'], errors='ignore').select_dtypes(include=[np.number]).values
y_all = df[['D', 'U']].values
N_USE = 3000


def evaluate_alpha_v_k(args):

    k_loss_dict = {}
    best_k = None
    alpha, v_threshold, k_feat = args
    print(f"\n===== alpha={alpha:.1f}, v_threshold={v_threshold}, top_k={k_feat} =====")

    X = X_all[:N_USE, :k_feat]
    y = y_all[:N_USE,:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    fold_losses = []

    # ==== 测试集评估 ====
    test_loss = 0
    obj_val = 0
    q_pred_val = 0
    cnt = 0
    reg = DecisionTreeRegressor(max_depth=deep)
    reg.fit(X_train, y_train)
    leaf_ids_train = reg.apply(X_train)
    for i, x_i in enumerate(X_test):
        d_i, u_i = y_test[i]

        # 获取目标样本落入的叶子节点ID
        leaf_ids = reg.apply(x_i.reshape(1, -1))

        matching_indices = np.where(leaf_ids_train == leaf_ids[0])[0]

        neighbor_y = y_train[matching_indices]

        d_vals = neighbor_y[:, 0]
        u_vals = neighbor_y[:, 1]
        k_nn = matching_indices.shape[0]
        weights = 1 / k_nn

        model = Model()
        model.setParam("OutputFlag", 0)
        q = model.addVar(lb=0.0, name="q")
        v = model.addVars(k_nn, lb=0.0)
        o = model.addVars(k_nn, lb=0.0)
        gamma = model.addVars(k_nn, vtype=GRB.BINARY)

        for j in range(k_nn):
            model.addConstr(v[j] >= d_vals[j] - q * u_vals[j])
            model.addConstr(o[j] >= q * u_vals[j] - d_vals[j])
            model.addConstr(b * v[j] + h * o[j] + M * gamma[j] >= v_threshold)
            model.addConstr(b * v[j] + h * o[j] + M * gamma[j] <= M + v_threshold)

        model.addConstr(sum(weights* gamma[j] for j in range(k_nn)) >= alpha)
        model.setObjective(sum(weights * (b * v[j] + h * o[j]) for j in range(k_nn)), GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            q_pred = q.X
            loss = b * max(d_i - q_pred * u_i, 0) + h * max(q_pred * u_i - d_i, 0)
            test_loss += loss

            # 计算真实目标函数值
            weighted_sum = sum(weights * (b * v[j].X + h * o[j].X) for j in range(k_nn))
            obj_val += weighted_sum

            # 统计q
            q_pred_val += q_pred
            cnt +=1
    if cnt > 0:
        test_loss /= cnt
        avg_obj_val = obj_val / cnt
        avg_q_pred = q_pred_val / cnt
    else:
        test_loss = np.nan
        avg_obj_val = np.nan
        avg_q_pred = np.nan

    best_result = {
        "alpha": alpha,
        "v_threshold": v_threshold,
        "best_test_loss": test_loss,
        'avg_obj_val': avg_obj_val,
        'avg_q_pred': avg_q_pred
    }

    all_k_results = [{
        "alpha": alpha,
        "v_threshold": v_threshold,
        "test_loss": test_loss if k_val == best_k else None
    } for k_val, train_loss in k_loss_dict.items()]

    return best_result, all_k_results


# ==== 主执行逻辑 ====
if __name__ == '__main__':
    param_grid = list(product(alpha_list, v_list, top_k_list))
    best_results, all_results = [], []

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = list(executor.map(evaluate_alpha_v_k, param_grid))

    for best, all_ks in futures:
        best_results.append(best)
        all_results.extend(all_ks)

    # pd.DataFrame(all_results).to_csv("CART/CART_loss_table_alpha_v_k_parallel2000.csv", index=False)
    pd.DataFrame(best_results).to_csv("CART/CART_best_k_per_alpha_v_parallel500-1500.csv", index=False)

    print("并发计算完成，结果已保存。")
