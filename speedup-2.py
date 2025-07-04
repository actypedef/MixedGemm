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
batch_size_powers = np.arange(6, 13)
x_labels = [f'$2^{{{p}}}$' for p in batch_size_powers]
x_indices = np.arange(len(x_labels))

# 数据源 1: RTX 5070 Ti Laptop
latency_data_5070ti = {
    'FP16 Baseline': np.array([0.139, 0.229, 0.232, 0.353, 0.644, 1.253, 2.529]),
    'FP8-TRT': np.array([0.048, 0.066, 0.109, 0.187, 0.450, 0.764, 1.534]),
    'W4A16-TRT': np.array([0.125, 0.145, 0.197, 0.302, 0.437, 0.850, 1.784]),
    'Atom (INT4)': np.array([0.517, 0.498, 0.758, 1.030, 1.937, 3.481, 6.929]),
    'MicroMix (Amortized)': np.array([0.092, 0.093, 0.097, 0.144, 0.262, 0.498, 0.953]),
    'MicroMix (Best Case)': np.array([0.089, 0.085, 0.088, 0.126, 0.234, 0.454, 0.856]),
    'MicroMix (Worst Case)': np.array([0.092, 0.093, 0.098, 0.148, 0.273, 0.517, 0.986]),
}

# 数据源 2: RTX 5090
latency_data_5090 = {
    'FP16 Baseline': np.array([0.026, 0.034, 0.067, 0.105, 0.200, 0.389, 0.766]),
    'FP8-TRT': np.array([0.022, 0.027, 0.035, 0.061, 0.098, 0.186, 0.382]),
    'W4A16-TRT': np.array([0.032, 0.039, 0.044, 0.068, 0.116, 0.240, 0.436]),
    'Atom (INT4)': np.array([0.253, 0.254, 0.250, 0.257, 0.497, 0.965, 1.681]),
    'MicroMix (Amortized)': np.array([0.092, 0.095, 0.100, 0.100, 0.087, 0.135, 0.242]),
    'MicroMix (Best Case)': np.array([0.094, 0.094, 0.093, 0.094, 0.084, 0.120, 0.209]),
    'MicroMix (Worst Case)': np.array([0.096, 0.098, 0.101, 0.103, 0.095, 0.139, 0.252]),
}

# 将所有数据源放入一个列表以便于迭代
all_latency_data = [latency_data_5070ti, latency_data_5090]
gpu_titles = ['Performance on RTX 5070 Ti Laptop', 'Performance on RTX 5090']

# --- 3. 绘图配置 (颜色、标记等) ---
colors = {
    'FP16 Baseline': '#8C8C8C', 'FP8-TRT': '#5DA5DA', 'W4A16-TRT': '#FAA43A',
    'Atom (INT4)': '#60BD68', 'MicroMix': '#F17CB0'
}
markers = {
    'FP16 Baseline': '', 'FP8-TRT': 's', 'W4A16-TRT': '^',
    'Atom (INT4)': 'D', 'MicroMix': '*'
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
        x_indices, speedup_data['MicroMix (Amortized)'], label='MicroMix (Amortized)',
        marker=markers['MicroMix'], color=colors['MicroMix'], linewidth=2.5, markersize=8
    )
    # 其他对比方法
    other_methods = ['FP8-TRT', 'W4A16-TRT', 'Atom (INT4)']
    for method in other_methods:
        ax.plot(
            x_indices, speedup_data[method], label=method,
            marker=markers[method], color=colors[method], linewidth=2
        )
        
    # --- 4c. 美化单个子图 ---
    ax.set_title(gpu_titles[i], fontsize=16, pad=15)
    ax.set_xlabel('Batch Size', fontsize=14, labelpad=10)
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
    labels.index('Atom (INT4)'),
    labels.index('MicroMix Speedup Range'),
    labels.index('MicroMix (Amortized)')
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