import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. 全局样式和字体设置 ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

# --- 2. 原始延迟数据定义 ---
# 横轴共享
seq_len_powers = np.arange(6, 13)
x_labels = [f'$2^{{{p}}}$' for p in seq_len_powers]
x_indices = np.arange(len(x_labels))

# 数据源 1: RTX 5070 Ti Laptop
latency_data_5070ti = {
    'FP16 Baseline': np.array([0.142, 0.278, 0.310, 0.533, 1.039, 1.873, 3.814]),
    'FP8-TRT': np.array([0.057, 0.092, 0.154, 0.297, 0.529, 1.159, 2.078]),
    'W4A16-TRT': np.array([0.174, 0.214, 0.269, 0.435, 0.704, 1.312, 2.526]),
    'MicroMix (Average)': np.array([0.095, 0.095, 0.110, 0.204, 0.372, 0.686, 1.313]),
    'MicroMix (Best Case)': np.array([0.094, 0.095, 0.096, 0.172, 0.315, 0.587, 1.126]),
    'MicroMix (Worst Case)': np.array([0.096, 0.097, 0.116, 0.210, 0.384, 0.725, 1.389]),
}

# 数据源 2: RTX 5090
latency_data_5090 = {
    'FP16 Baseline': np.array([0.051, 0.049, 0.073, 0.132, 0.250, 0.480, 0.948]),
    'FP8-TRT': np.array([0.026, 0.034, 0.044, 0.075, 0.142, 0.283, 0.554]),
    'W4A16-TRT': np.array([0.043, 0.050, 0.064, 0.094, 0.181, 0.340, 0.657]),
    'MicroMix (Average)': np.array([0.092, 0.095, 0.092, 0.093, 0.096, 0.177, 0.359]),
    'MicroMix (Best Case)': np.array([0.090, 0.090, 0.089, 0.093, 0.092, 0.151, 0.298]),
    'MicroMix (Worst Case)': np.array([0.096, 0.096, 0.097, 0.098, 0.101, 0.187, 0.380]),
}

# 将所有数据源放入一个列表以便于迭代
all_latency_data = [latency_data_5070ti, latency_data_5090]
gpu_titles = ['Performance on RTX 5070 Ti Laptop', 'Performance on RTX 5090']

# --- 3. 绘图配置 (颜色、标记等) ---
colors = {
    'FP16 Baseline': '#8C8C8C', 'FP8-TRT': '#5DA5DA', 'W4A16-TRT': '#FAA43A',
    'MicroMix': '#F17CB0'
}
markers = {
    'FP16 Baseline': '', 'FP8-TRT': 's', 'W4A16-TRT': '^',
    'MicroMix': '*'
}

# --- 4. 开始绘图 ---
# 创建一个 1x2 的子图网格，并设置一个更宽的尺寸以容纳两个图
fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), sharey=True) # sharey=True 使Y轴刻度对齐

# 设置整个图表的大标题
fig.suptitle('Kernel Performance Speedup Relative to FP16 Baseline', fontsize=20, fontweight='bold')

# 循环绘制每个子图
for i, ax in enumerate(axes):
    latency_data = all_latency_data[i]
    
    # --- 4a. 将延迟数据转换为加速比 ---
    speedup_data = {}
    fp16_baseline_latencies = latency_data['FP16 Baseline']
    for method, latencies in latency_data.items():
        with np.errstate(divide='ignore', invalid='ignore'):
            speedup_data[method] = np.divide(fp16_baseline_latencies, latencies)
    
    # --- 4b. 在对应的子图(ax)上绘制数据 ---
    # MicroMix 加速比范围
    ax.fill_between(
        x_indices, speedup_data['MicroMix (Best Case)'], speedup_data['MicroMix (Worst Case)'],
        color=colors['MicroMix'], alpha=0.2, label='MicroMix Speedup Range'
    )
    # FP16 基线 (y=1)
    ax.plot(
        x_indices, speedup_data['FP16 Baseline'], label='FP16 Baseline',
        color=colors['FP16 Baseline'], linestyle='--', linewidth=2
    )
    # MicroMix 均摊性能
    ax.plot(
        x_indices, speedup_data['MicroMix (Average)'], label='MicroMix (Average)',
        marker=markers['MicroMix'], color=colors['MicroMix'], linewidth=2.5, markersize=8
    )
    # 其他对比方法
    other_methods = ['FP8-TRT', 'W4A16-TRT']
    for method in other_methods:
        ax.plot(
            x_indices, speedup_data[method], label=method,
            marker=markers[method], color=colors[method], linewidth=2
        )
        
    # --- 4c. 美化单个子图 ---
    ax.set_title(gpu_titles[i], fontsize=16, pad=15)
    ax.set_xlabel('Sequence Lenght', fontsize=14, labelpad=10)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(bottom=0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# 为第一个子图（左图）设置Y轴标签
axes[0].set_ylabel('Speedup (vs. FP16)', fontsize=14, labelpad=10)

# --- 5. 创建并放置共享图例 ---
handles, labels = axes[0].get_legend_handles_labels() # 从任意一个子图获取图例句柄和标签
# 自定义图例顺序
order = [
    labels.index('FP16 Baseline'),
    labels.index('FP8-TRT'),
    labels.index('W4A16-TRT'),
    labels.index('MicroMix Speedup Range'),
    labels.index('MicroMix (Average)')
]

# 在图表下方居中放置图例
fig.legend(
    [handles[idx] for idx in order], 
    [labels[idx] for idx in order],
    loc='lower center',         # 定位到下方中央
    bbox_to_anchor=(0.5, -0.01), # 微调位置，(0.5, 0)是底部边缘，小于0则向下移动
    ncol=len(handles),          # 将所有图例项放在一行
    frameon=False               # 去掉图例边框
)

# --- 6. 调整整体布局 ---
# 调整布局，为大标题和底部图例留出空间
plt.tight_layout(rect=[0, 0.1, 1, 0.93])

# 保存图像 (可选)
# plt.savefig("kernel_speedup_comparison_dual_gpu.png", dpi=300, bbox_inches='tight')
# plt.savefig("kernel_speedup_comparison_dual_gpu.pdf", bbox_inches='tight')

# 显示图像
plt.show()