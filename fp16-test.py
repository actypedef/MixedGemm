import torch
from torch.nn import functional as F

# import pycuda.driver as cuda

# class L2Flush:
#     def __init__(self):
#         cuda.init()
#         self.device = cuda.Device(0)
#         self.context = self.device.make_context()
#         self.m_l2_size = self.device.get_attribute(cuda.device_attribute.L2_CACHE_SIZE)
#         if self.m_l2_size > 0:
#             print("device L2 cache size: %d MB" % (self.m_l2_size / 1024 / 1024))
#             self.m_l2_buffer = cuda.mem_alloc(self.m_l2_size)

#     def __del__(self):
#         if self.m_l2_buffer:
#             self.m_l2_buffer.free()

#         # Detach the CUDA context when the object is deleted
#         self.context.pop()

#     def flush(self, stream):
#         if self.m_l2_size > 0:
#             self.context.push()
#             cuda.memset_d8_async(self.m_l2_buffer, 0, self.m_l2_size, stream)
#             self.context.pop()
# l2Flusher = L2Flush()
# l2Flusher.flush(cuda.Stream(0))

test_M_pool = [64, 128, 256, 512, 1024, 2048, 4096]


@torch.no_grad()
def test_quant_linear_a16_w16(M, N ,K) -> float:
    weight = torch.rand(K, N, dtype=torch.float16).cuda()
    x = torch.rand(M, K, dtype=torch.float16).cuda()

    elapsed_time_ms = 0
    iterations = 30
    for _ in range(iterations):
        torch.cuda.synchronize()
        y = F.linear(x, weight)
        torch.cuda.synchronize()

    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        y = F.linear(x, weight)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms += start_event.elapsed_time(end_event)

        # l2Flusher.flush(cuda.Stream(0))
    total_ops = M * N * K * 2 * iterations
    gflops = total_ops / elapsed_time_ms / 10**9
    return gflops, elapsed_time_ms / iterations

a16w16_flops = []
a16w16_times = []
for m in test_M_pool:
    gflops, elapsed_time_ms = test_quant_linear_a16_w16(m, 5120, 5120)
    a16w16_flops.append(gflops)
    a16w16_times.append(elapsed_time_ms)

print(a16w16_times)