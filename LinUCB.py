import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def DisjointLinUCB(data_array, alpha, num_arms=10):
    trials = data_array.shape[0]  # 试验次数（数据的行数）
    num_features = data_array.shape[1] - 2  # 特征的数量（去掉arm和reward列）

    # 初始化 A, b, 和 theta
    A = [np.eye(num_features) for _ in range(num_arms)]  # 每个臂的 A 初始化为单位矩阵
    b = [np.zeros((num_features, 1)) for _ in range(num_arms)]  # 每个臂的 b 初始化为零向量
    theta = [np.zeros((num_features, 1)) for _ in range(num_arms)]  # 每个臂的 theta 初始化为零向量

    total_payoff = 0  # 总奖励

    for t in range(trials):
        arm = int(data_array[t, 0]) - 1  # 获取当前臂的索引
        payoff = data_array[t, 1]  # 当前试验的奖励
        x_t = np.expand_dims(data_array[t, 2:], axis=1)  # 获取当前臂的特征并转为列向量

        # 计算每个臂的 p 值（探索项 + 利用项）
        p = []
        for a in range(num_arms):
            A_inv = np.linalg.inv(A[a])  # 计算 A 的逆
            theta_a = np.matmul(A_inv, b[a])  # 计算 theta
            exploration_term = alpha * np.sqrt(np.matmul(np.matmul(x_t.T, A_inv), x_t))  # 探索项
            p.append(np.matmul(theta_a.T, x_t) + exploration_term)  # p = 利用项 + 探索项

        # 选择具有最大 p 值的臂
        selected_arm = np.argmax(p)

        # 更新 A, b 和 theta
        if selected_arm == arm:
            A[selected_arm] += np.matmul(x_t, x_t.T)  # 更新 A
            b[selected_arm] += payoff * x_t  # 更新 b

            # 更新 theta
            theta[selected_arm] = np.linalg.inv(A[selected_arm]) @ b[selected_arm]
            total_payoff += payoff

    return total_payoff / trials  # 返回平均CTR（点击率）

def HybridLinUCB(data_array, alpha,num_arms=10):

    num_trials = data_array.shape[0]  # 总试验次数（行数）
    num_features = data_array.shape[1] - 2  # 特征数量（去掉 arm_id 和 reward 列）

    # 初始化 A_0, b_0
    A_0 = np.eye(num_features)  # 全局 A_0 是单位矩阵
    b_0 = np.zeros((num_features, 1))  # 全局 b_0 是零向量

    # 初始化每个臂的 A_a, B_a, b_a
    A = [np.eye(num_features) for _ in range(num_arms)]  # 每个臂的 A 初始化为单位矩阵
    B = [np.zeros((num_features, num_features)) for _ in range(num_arms)]  # 每个臂的 B 初始化为零矩阵
    b = [np.zeros((num_features, 1)) for _ in range(num_arms)]  # 每个臂的 b 初始化为零向量

    total_payoff = 0  # 累计奖励

    for t in range(num_trials):
        # 观察所有臂的特征数据
        arm_id = int(data_array[t, 0]) - 1  # 获取当前臂的编号（从 0 开始）
        reward = data_array[t, 1]  # 当前试验的奖励
        x_t = np.expand_dims(data_array[t, 2:], axis=1)  # 获取当前臂的特征并转为列向量

        # 计算 beta_hat = A_0^-1 * b_0
        beta_hat = np.linalg.inv(A_0) @ b_0

        # 计算每个臂的 p(t, a)
        p = np.zeros(num_arms)
        for a in range(num_arms):
            if np.array_equal(A[a], np.eye(num_features)):  # 如果该臂是新臂，则初始化
                A[a] = np.eye(num_features)
                B[a] = np.zeros((num_features, num_features))
                b[a] = np.zeros((num_features, 1))

            # 计算 theta_a = A_a^-1 * (b_a - B_a * beta_hat)
            theta_a = np.linalg.inv(A[a]) @ (b[a] - B[a] @ beta_hat)

            # 计算 s(t, a)
            z_t_a = x_t  # z(t, a) 是当前臂的特征
            term1 = z_t_a.T @ np.linalg.inv(A_0) @ z_t_a
            term2 = 2 * z_t_a.T @ np.linalg.inv(A_0) @ B[a].T @ np.linalg.inv(A[a]) @ x_t
            term3 = x_t.T @ np.linalg.inv(A[a]) @ x_t
            term4 = x_t.T @ np.linalg.inv(A[a]) @ B[a] @ np.linalg.inv(A_0) @ B[a].T @ np.linalg.inv(A[a]) @ x_t
            s_t_a = term1 - term2 + term3 + term4

            # 计算 p(t, a)
            p[a] = z_t_a.T @ beta_hat + x_t.T @ theta_a + alpha * np.sqrt(s_t_a)

        # 选择最大 p(t, a) 的臂
        selected_arm = np.argmax(p)

        # 仅当选择的臂与当前臂一致时才更新
        if selected_arm == arm_id:
            # **第一次更新 A_0 和 b_0**
            A_0 += B[selected_arm].T @ np.linalg.inv(A[selected_arm]) @ B[selected_arm]
            b_0 += B[selected_arm].T @ np.linalg.inv(A[selected_arm]) @ b[selected_arm]

            # 更新选定臂的 A, B, b
            A[selected_arm] += x_t @ x_t.T
            B[selected_arm] += x_t @ x_t.T
            b[selected_arm] += reward * x_t

            # **第二次更新 A_0 和 b_0**
            z_t_a = np.expand_dims(data_array[t, 2:], axis=1)  # 确保 z_t_a 是列向量
            term1 = z_t_a @ z_t_a.T  # term1 是矩阵，形状为 (num_features, num_features)
            term2 = B[selected_arm].T @ np.linalg.inv(A[selected_arm]) @ B[selected_arm]
            term3 = B[selected_arm].T @ np.linalg.inv(A[selected_arm]) @ b[selected_arm]

            # 计算 term3 时将其转换为列向量
            #term3 = np.reshape(term3, (num_features, 1))  # 将 term3 转换为列向量

            # 现在保证所有更新项的形状兼容
            A_0 += term1-term2
            b_0 += reward*z_t_a-term3

            total_payoff += reward  # 累计奖励

    avg_ctr = total_payoff / num_trials  # 返回平均CTR
    return avg_ctr


def UCB(data_array, alpha=1.0, num_arms=10):

    trials = data_array.shape[0]  # 试验次数（数据的行数）

    # 初始化每个臂的选择次数和奖励
    counts = np.zeros(num_arms, dtype=int)  # 每个臂的选择次数
    rewards = np.zeros(num_arms)  # 每个臂的总奖励

    total_payoff = 0  # 总奖励

    # 记录选择的臂以及对应的 UCB 值
    for t in range(1, trials + 1):  # 从 1 开始，避免除以零
        arm_id = int(data_array[t - 1, 0]) - 1  # 获取当前臂的索引（-1因为数据是从1开始的）
        payoff = data_array[t - 1, 1]  # 当前试验的奖励

        # 计算每个臂的 UCB 值
        p = []
        for a in range(num_arms):
            if counts[a] == 0:  # 如果该臂未被选择过，则直接选择该臂
                p.append(np.inf)  # 给未选择的臂一个极大值
            else:
                # 计算平均奖励
                mean_reward = rewards[a] / counts[a]
                # 计算探索项：sqrt( (2 * log(t)) / counts[a] )
                exploration_term = np.sqrt((2 * np.log(t)) / counts[a])
                # UCB 公式
                p.append(mean_reward + alpha * exploration_term)

        # 选择具有最大 UCB 值的臂
        selected_arm = np.argmax(p)

        if selected_arm == arm_id:
           counts[selected_arm] += 1
           rewards[selected_arm] += payoff

           total_payoff += payoff  # 累计奖励

    # 计算平均 CTR
    avg_ctr = total_payoff / trials
    return avg_ctr

def epsilon_greedy(data_array, epsilon, num_arms=10):

    trials = data_array.shape[0]  # 试验次数（数据的行数）

    # 初始化每个臂的选择次数和奖励
    counts = np.zeros(num_arms, dtype=int)  # 每个臂的选择次数
    rewards = np.zeros(num_arms)  # 每个臂的总奖励

    total_payoff = 0  # 总奖励

    for t in range(trials):
        arm_id = int(data_array[t, 0]) - 1  # 获取当前臂的索引（-1因为数据是从1开始的）
        payoff = data_array[t, 1]  # 当前试验的奖励

        # 探索或利用
        if np.random.rand() < epsilon:
            # 探索：随机选择一个臂
            selected_arm = np.random.choice(num_arms)
        else:
            # 利用：选择当前平均奖励最大的臂
            avg_rewards = rewards / (counts + 1e-5)  # 加一个小的常数避免除以零
            selected_arm = np.argmax(avg_rewards)

        if selected_arm == arm_id:
           counts[selected_arm] += 1
           rewards[selected_arm] += payoff

           total_payoff += payoff  # 累计奖励

    # 计算平均 CTR
    avg_ctr = total_payoff / trials
    return avg_ctr  # 返回平均CTR（点击率）


def run_UCB(data_array, alphas):
    ctr_values = []
    for alpha in alphas:
        ctr = UCB(data_array, alpha=alpha, num_arms=10)  # 使用 UCB 算法
        ctr_values.append(ctr)
        print(f"UCB Average CTR for alpha={alpha}: {ctr}")
    return ctr_values

def run_epsilon_greedy(data_array, epsilons):
    ctr_values = []
    for epsilon in epsilons:
        ctr = epsilon_greedy(data_array, epsilon=epsilon, num_arms=10)  # 使用 epsilon-greedy 算法
        ctr_values.append(ctr)
        print(f"epsilon_greedy Average CTR for epsilon={epsilon}: {ctr}")
    return ctr_values

def run_DisjointLinUCB(data_array, alphas):
    ctr_values = []
    for alpha in alphas:
        ctr = DisjointLinUCB(data_array, alpha=alpha, num_arms=10)  # 使用 DisjointLinUCB 算法
        ctr_values.append(ctr)
        print(f"DisjointLinUCB Average CTR for alpha={alpha}: {ctr}")
    return ctr_values

def run_HybridLinUCB(data_array, alphas):
    ctr_values = []
    for alpha in alphas:
        ctr = HybridLinUCB(data_array, alpha=alpha)  # 使用 HybridLinUCB 算法
        ctr_values.append(ctr)
        print(f"HybridLinUCB Average CTR for alpha={alpha}: {ctr}")
    return ctr_values

data_path=r'dataset.txt'
data = np.loadtxt(data_path)
alphas = np.arange(0, 1.01, 0.05)

df = pd.read_csv('ctr_data_with_alpha.csv')

# 提取 alpha 和各个算法的CTR数据
alphas = df['Alpha'].values
ucb_ctr = df['UCB'].values
epsilon_greedy_ctr = df['epsilon-greedy'].values
disjoint_linucb_ctr = df['DisjointLinUCB'].values
hybrid_linucb_ctr = df['HybridLinUCB'].values

# 绘制折线图
plt.figure(figsize=(10, 6))

# UCB算法折线
plt.plot(alphas, ucb_ctr, label="UCB", marker='o')
# epsilon-greedy算法折线
plt.plot(alphas, epsilon_greedy_ctr, label="epsilon-greedy", marker='s')
# DisjointLinUCB算法折线
plt.plot(alphas, disjoint_linucb_ctr, label="DisjointLinUCB", marker='^')
# HybridLinUCB算法折线
plt.plot(alphas, hybrid_linucb_ctr, label="HybridLinUCB", marker='d')

# 设置标题、坐标轴标签和图例
plt.title('CTR vs Alpha for Different Algorithms')
plt.xlabel('Alpha')
plt.ylabel('Average CTR')
plt.legend()

# 显示图形
plt.grid(True)
plt.show()


