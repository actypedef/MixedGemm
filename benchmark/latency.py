import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. 全局样式和字体设置 (符合学术要求) ---
# 使用 Seaborn 提升美感
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif' # 可以指定更具体的字体，如 'Times New Roman'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150 # 提高图像分辨率

# --- 2. 数据定义 (易于扩展) ---
# 在这里替换成您的真实数据

# 横轴: Batch Size (2^6 to 2^12)
batch_size_powers = np.arange(6, 13)
x_labels = [f'$2^{{{p}}}$' for p in batch_size_powers]
# 用于绘图的均匀间隔的 x 轴坐标
x_indices = np.arange(len(x_labels))


# latency_data = {
#     # 请注意，为了与之前的绘图代码兼容，方法名称已进行调整
#     # FP16 Baseline <-> FP16
#     'FP16 Baseline': np.array([0.139, 0.229, 0.232, 0.353, 0.644, 1.253, 2.529]),
    
#     # FP8-TRT <-> FP8-TRT
#     'FP8-TRT': np.array([0.048, 0.066, 0.109, 0.187, 0.450, 0.764, 1.534]),
    
#     # W4A16-TRT <-> W4A16-TRT
#     'W4A16-TRT': np.array([0.125, 0.145, 0.197, 0.302, 0.437, 0.850, 1.784]),
    
#     # Atom (INT4) <-> Atom(INT4)
#     'Atom (INT4)': np.array([0.517, 0.498, 0.758, 1.030, 1.937, 3.481, 6.929]),
    
#     # MicroMix (Amortized) <-> MicroMix(Ave.)
#     'MicroMix (Amortized)': np.array([0.092, 0.093, 0.097, 0.144, 0.262, 0.498, 0.953]),
    
#     # MicroMix (Best Case) <-> MicroMix(Best)
#     'MicroMix (Best Case)': np.array([0.089, 0.085, 0.088, 0.126, 0.234, 0.454, 0.856]),
    
#     # MicroMix (Worst Case) <-> MicroMix(Worst)
#     'MicroMix (Worst Case)': np.array([0.092, 0.093, 0.098, 0.148, 0.273, 0.517, 0.986]),
# }

latency_data = {
    # FP16 Baseline <-> FP16
    'FP16 Baseline': np.array([0.026, 0.034, 0.067, 0.105, 0.200, 0.389, 0.766]),
    
    # FP8-TRT <-> FP8-TRT
    'FP8-TRT': np.array([0.022, 0.027, 0.035, 0.061, 0.098, 0.186, 0.382]),
    
    # W4A16-TRT <-> W4A16-TRT
    'W4A16-TRT': np.array([0.032, 0.039, 0.044, 0.068, 0.116, 0.240, 0.436]),
    
    # Atom (INT4) <-> Atom(INT4)
    'Atom (INT4)': np.array([0.253, 0.254, 0.250, 0.257, 0.497, 0.965, 1.681]),
    
    # MicroMix (Amortized) <-> MicroMix(Ave.)
    'MicroMix (Amortized)': np.array([0.092, 0.095, 0.100, 0.100, 0.087, 0.135, 0.242]),
    
    # MicroMix (Best Case) <-> MicroMix(Best)
    'MicroMix (Best Case)': np.array([0.094, 0.094, 0.093, 0.094, 0.084, 0.120, 0.209]),
    
    # MicroMix (Worst Case) <-> MicroMix(Worst)
    'MicroMix (Worst Case)': np.array([0.096, 0.098, 0.101, 0.103, 0.095, 0.139, 0.252]),
}

# --- 3. 绘图配置 (颜色、标记等) ---
# 高级感配色方案 (灵感来自 Tableau 或 ColorBrewer)
colors = {
    'FP16 Baseline': '#8C8C8C',      # 灰色
    'FP8-TRT': '#5DA5DA',            # 蓝色
    'W4A16-TRT': '#FAA43A',          # 橙色
    'Atom (INT4)': '#60BD68',         # 绿色
    'MicroMix': '#F17CB0'            # 粉色
}

# 标记样式，确保黑白打印时也能区分
markers = {
    'FP16 Baseline': 'o',  # 圆圈
    'FP8-TRT': 's',      # 方块
    'W4A16-TRT': '^',    # 三角
    'Atom (INT4)': 'D',    # 菱形
    'MicroMix': '*'      # 星形
}

# --- 4. 开始绘图 ---
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制 MicroMix 的性能范围
ax.fill_between(
    x_indices,
    latency_data['MicroMix (Best Case)'],
    latency_data['MicroMix (Worst Case)'],
    color=colors['MicroMix'],
    alpha=0.2, # 使用透明度
    label='MicroMix Performance Range (Best to Worst)'
)

# 绘制所有方法的性能折线
# 绘制 MicroMix 均摊性能
ax.plot(
    x_indices,
    latency_data['MicroMix (Amortized)'],
    label='MicroMix (Amortized)',
    marker=markers['MicroMix'],
    color=colors['MicroMix'],
    linewidth=2.5, # 让主角线更突出
    markersize=8
)

# 绘制其他对比方法
other_methods = ['FP16 Baseline', 'FP8-TRT', 'W4A16-TRT', 'Atom (INT4)']
for method in other_methods:
    ax.plot(
        x_indices,
        latency_data[method],
        label=method,
        marker=markers[method],
        color=colors[method],
        linewidth=2
    )

# --- 5. 图表美化与信息标注 ---

# 设置标题和坐标轴标签
ax.set_title('Performance Comparison of Different CUDA Kernels', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Batch Size', fontsize=14, labelpad=10)
ax.set_ylabel('Latency (ms)', fontsize=14, labelpad=10)

# 设置横轴刻度标签
ax.set_xticks(x_indices)
ax.set_xticklabels(x_labels)

# 设置纵轴从0开始
ax.set_ylim(bottom=0)

# 添加图例
# 重新排列图例顺序，让 MicroMix 的两个条目在一起
handles, labels = ax.get_legend_handles_labels()
# 自定义顺序: MicroMix范围, MicroMix均摊, 然后是其他
order = [labels.index('MicroMix Performance Range (Best to Worst)'), labels.index('MicroMix (Amortized)')]
order += [labels.index(m) for m in other_methods]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')

# 优化网格线
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# 调整布局防止标签被裁剪
plt.tight_layout()

# 保存图像 (可选)
# plt.savefig("kernel_performance_comparison.png", dpi=300, bbox_inches='tight')
# plt.savefig("kernel_performance_comparison.pdf", bbox_inches='tight')

# 显示图像
plt.show()