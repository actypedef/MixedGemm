import sys
sys.path.append('build/')
import torch
import mixedgemm  # 确保 mixedgemm.so 位于 Python 的模块搜索路径中

# 设置矩阵维度
M, N, K = 1024, 4096, 4096
group = 32
KN, KS, KO = 2560, 1408, 128

# 创建输入张量
AN = torch.randint(0, 1, (M, KN // 2), dtype=torch.uint8, device='cuda')
BN = torch.randint(0, 1, (N, KN // 2), dtype=torch.uint8, device='cuda')
AS = torch.randint(0, 1, (M, KS // 4 * 3), dtype=torch.uint8, device='cuda')
BS = torch.randint(0, 1, (N, KS // 2), dtype=torch.uint8, device='cuda')
AO = torch.randint(0, 1, (M, KO), dtype=torch.uint8, device='cuda')
BO = torch.randint(0, 1, (N, KO // 2), dtype=torch.uint8, device='cuda')

# 创建缩放因子张量
SFAN = torch.randint(1, 2, (1, M * KN // group), dtype=torch.uint8, device='cuda')
SFBN = torch.randint(1, 2, (1, N * KN // group), dtype=torch.uint8, device='cuda')
SFAS = torch.randint(1, 2, (1, M * KS // group), dtype=torch.uint8, device='cuda')
SFBS = torch.randint(1, 2, (1, N * KS // group), dtype=torch.uint8, device='cuda')
SFAO = torch.randint(1, 2, (1, M * KO // group), dtype=torch.uint8, device='cuda')
SFBO = torch.randint(1, 2, (1, N * KO // group), dtype=torch.uint8, device='cuda')

# 调用 C++ 扩展中的 matmul 函数
C = mixedgemm.matmul(AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO)

print("输出张量 C 的形状:", C.shape)
print("输出张量 C 的数据类型:", C.dtype)
# 计算平均值
mean_value = torch.mean(C)

# 计算方差
variance_value = torch.var(C)

print(f"平均值: {mean_value.item():.6f}")
print(f"方差: {variance_value.item():.6f}")
