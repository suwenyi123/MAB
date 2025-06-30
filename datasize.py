from LinUCB import DisjointLinUCB,HybridLinUCB,UCB,epsilon_greedy
import numpy as np
import matplotlib.pyplot as plt

data_path=r'dataset.txt'
data = np.loadtxt(data_path)
disjoint_ctr=[]
hybrid_ctr=[]
ucb_ctr=[]
epsgreedy_ctr=[]

percentages = [1.0, 0.3, 0.2, 0.1, 0.05]
for percentage in percentages:
    subset_data = data[:int(len(data) * percentage)]  # 按比例获取数据集的子集
    ctr = DisjointLinUCB(subset_data, 0.05, num_arms=10)
    disjoint_ctr.append(ctr)
    ctr = HybridLinUCB(subset_data, 0.05, num_arms=10)
    hybrid_ctr.append(ctr)
    ctr = UCB(subset_data, 0.05, num_arms=10)
    ucb_ctr.append(ctr)
    ctr = epsilon_greedy(subset_data, 0.1, num_arms=10)
    epsgreedy_ctr.append(ctr)

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 8))

# X轴位置
x_pos = np.arange(len(percentages))

# 每种算法的颜色

# 每种算法的宽度
bar_width = 0.2
colors = ['#4C9EFF', '#70E1B7', '#FFDC61', '#B88CFF']
# 绘制每种算法的柱状图
ax.bar(x_pos - 1.5 * bar_width, disjoint_ctr, bar_width, label='DisjointLinUCB',color=colors[0])
ax.bar(x_pos - 0.5 * bar_width, hybrid_ctr, bar_width, label='HybridLinUCB',color=colors[1])
ax.bar(x_pos + 0.5 * bar_width, ucb_ctr, bar_width, label='UCB', color=colors[2])
ax.bar(x_pos + 1.5 * bar_width, epsgreedy_ctr, bar_width, label='epsilon-greedy', color=colors[3])

# 设置图表的标签
ax.set_xlabel('Data Set Size (%)', fontsize=14)
ax.set_ylabel('Average CTR', fontsize=14)
ax.set_title('CTR Comparison for Different Data Set Sizes', fontsize=16)

# 设置X轴标签（百分比）
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{int(p * 100)}%" for p in percentages])  # 将小数转为百分比格式

# 设置图例
ax.legend()

# 显示图形
plt.tight_layout()
plt.show()