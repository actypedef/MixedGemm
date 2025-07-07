import sys
sys.path.append('build/')
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import mixedgemm  

for i in range(10):
    M, N, K = 2048, 4096, 4096
    group = 32
    KN, KS, KO = 2048, 1024, 1024


    signs = (torch.randint(0, 2, (M, K), device='cuda', dtype=torch.bfloat16) * 2 - 1)
    X = torch.rand(M, K, dtype=torch.bfloat16, device='cuda') * 3
    X[:, -KS:] = torch.rand(M, KS, dtype=torch.bfloat16, device='cuda') * 14 + 14
    X[:, -KN:] = torch.rand(M, KN, dtype=torch.bfloat16, device='cuda') * 256 + 256
    X = X * signs
    # W = torch.rand(N, K, dtype=torch.bfloat16, device='cuda') * 13
    W = torch.eye(K, dtype=torch.bfloat16, device='cuda') * 1
    NormW = torch.rand(K, dtype=torch.bfloat16, device='cuda')
    # NormW = torch.ones(K, dtype=torch.bfloat16, device='cuda')
    reorder_index = torch.arange(K, dtype=torch.int16, device='cuda') 

    WT = W.t().clone()
    AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.rmsnorm_quantize_x(X, NormW, 1e-6, reorder_index, KN, KS, KO)
    BN, BS, BO, SFBN, SFBS, SFBO = mixedgemm.reorder_quantize_w(W, reorder_index, KN, KS, KO)

    C = mixedgemm.matmul(AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO)

    
    D = torch.matmul(F.rms_norm(X, (4096,), weight=NormW, eps=1e-6), WT)

    mean_value = torch.mean(C)

    variance_value = torch.var(C)

    mean_valued = torch.mean(D)

    variance_valued = torch.var(D)

    # E = C - D
    mse_loss_fn = torch.nn.MSELoss()
    mse_alt = mse_loss_fn(C, D)
    # variance_error = torch.var(E)

    print(f"平均值c: {mean_value.item():.6f}")
    print(f"方差c: {variance_value.item():.6f}")
    print(f"平均值d: {mean_valued.item():.6f}")
    print(f"方差d: {variance_valued.item():.6f}")
    print(f"valueC:{C.flatten()[:10]}...{C.flatten()[-10:]}")
    print(f"valueD:{D.flatten()[:10]}...{D.flatten()[-10:]}")
    print(f"误差E: {mse_alt.item() / 1e6:.6f}")
    print(f"finish {i}")
    time.sleep(1)