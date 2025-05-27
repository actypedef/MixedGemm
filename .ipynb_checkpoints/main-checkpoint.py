import sys
sys.path.append('build/')
import torch
import mixedgemm  

M, N, K = 1024, 4096, 4096
group = 32
KN, KS, KO = 2560, 1408, 128


# X = torch.ones(M, K, dtype=torch.bfloat16, device='cuda') * 1
# W = torch.ones(N, K, dtype=torch.bfloat16, device='cuda') * 1
X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
# reorder_index = torch.randperm(K, dtype=torch.int16, device='cuda')
reorder_index = torch.arange(K, dtype=torch.int16, device='cuda') 
AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(X, reorder_index, KN, KS, KO)
BN, BS, BO, SFBN, SFBS, SFBO = mixedgemm.reorder_quantize_w(W, reorder_index, KN, KS, KO)

print("--- Outputs from reorder_quantize_x ---")
outputs_x = {"AN": AN, "AS": AS, "AO": AO, "SFAN": SFAN, "SFAS": SFAS, "SFAO": SFAO}
for name, tensor_val in outputs_x.items():
    print(f"{name}: shape={tensor_val.shape}, dtype={tensor_val.dtype}")
    if torch.is_floating_point(tensor_val) or tensor_val.dtype == torch.bfloat16: # Only check float/bf16 for nan/inf
        print(f"  {name} has nan: {torch.isnan(tensor_val.float()).any().item()}") # Convert to float32 for isnan if needed
        print(f"  {name} has inf: {torch.isinf(tensor_val.float()).any().item()}")
        # print(f"  {name} sample (float32 view): {tensor_val.float().flatten()[:10]}") # View as float32 to see values
    else: # For uint8 tensors, print raw values
        print(f"  {name} sample (uint8): {tensor_val.flatten()[:10]}")
print("--- Outputs from reorder_quantize_w ---")
outputs_x = {"BN": BN, "BS": BS, "BO": BO, "SFBN": SFBN, "SFBS": SFBS, "SFBO": SFBO}
for name, tensor_val in outputs_x.items():
    print(f"{name}: shape={tensor_val.shape}, dtype={tensor_val.dtype}")
    if torch.is_floating_point(tensor_val) or tensor_val.dtype == torch.bfloat16: # Only check float/bf16 for nan/inf
        print(f"  {name} has nan: {torch.isnan(tensor_val.float()).any().item()}") # Convert to float32 for isnan if needed
        print(f"  {name} has inf: {torch.isinf(tensor_val.float()).any().item()}")
        # print(f"  {name} sample (float32 view): {tensor_val.float().flatten()[:10]}") # View as float32 to see values
    else: # For uint8 tensors, print raw values
        print(f"  {name} sample (uint8): {tensor_val.flatten()[:10]}")

C = mixedgemm.matmul(AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO)

print("输出张量 C 的形状:", C.shape)
print("输出张量 C 的数据类型:", C.dtype)

mean_value = torch.mean(C)

variance_value = torch.var(C)

print(f"平均值: {mean_value.item():.6f}")
print(f"方差: {variance_value.item():.6f}")
print(f"value:{C.flatten()[:20]}")