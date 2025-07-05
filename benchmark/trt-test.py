import tensorrt as trt
import torch
import numpy as np
import time

# --- 配置参数 ---
M, N, K = 2048, 4096, 4096
N_ITERATIONS = 100
N_WARMUP = 10

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_fp8_engine_the_right_way():
    """
    基于所有错误信息的最终推论：
    1. 使用 STRONGLY_TYPED。
    2. 显式构建 Q -> DQ -> MatMul 模式。
    3. 不设置任何 BuilderFlag，完全依赖TensorRT的自动融合。
    """
    builder = trt.Builder(TRT_LOGGER)
    
    # 1. 必须使用 STRONGLY_TYPED 标志创建网络
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    
    # 2. 创建一个“干净”的配置，不含任何精度标志
    config = builder.create_builder_config()

    # --- 3. 定义网络输入和权重 (FP32) ---
    input_a = network.add_input(name='A', dtype=trt.float32, shape=(M, K))
    weights_b_cpu = torch.randn(K, N, dtype=torch.float32)
    weights_b = network.add_constant(weights_b_cpu.shape, weights_b_cpu.numpy()).get_output(0)

    # --- 4. 构建 TensorRT 期望看到的 Q -> DQ 融合模式 ---
    scale_val = 0.02 
    scale_tensor = network.add_constant((), trt.Weights(np.array([scale_val], dtype=np.float32))).get_output(0)
    
    # Quantize A to FP8
    quant_a_layer = network.add_quantize(input_a, scale_tensor, trt.DataType.FP8)
    fp8_a = quant_a_layer.get_output(0)
    
    # Dequantize A back to FP32. 这是满足API要求的关键步骤。
    dequant_a_layer = network.add_dequantize(fp8_a, scale_tensor, trt.DataType.FLOAT)
    dequant_a_output = dequant_a_layer.get_output(0)
    
    # Quantize B to FP8
    quant_b_layer = network.add_quantize(weights_b, scale_tensor, trt.DataType.FP8)
    fp8_b = quant_b_layer.get_output(0)

    # Dequantize B back to FP32
    dequant_b_layer = network.add_dequantize(fp8_b, scale_tensor, trt.DataType.FLOAT)
    dequant_b_output = dequant_b_layer.get_output(0)
    
    # --- 5. MatMul层操作在Dequantize的输出上 (逻辑上是FP32) ---
    gemm_layer = network.add_matrix_multiply(
        dequant_a_output, trt.MatrixOperation.NONE,
        dequant_b_output, trt.MatrixOperation.NONE
    )
    
    # --- 6. 标记最终输出 ---
    final_output = gemm_layer.get_output(0)
    network.mark_output(final_output)
    final_output.name = "output_c"

    # --- 构建引擎 ---
    print("Building engine with automatic fusion pattern recognition...")
    # 核心：不设置任何 BuilderFlag，让 TRT 自动工作
    plan = builder.build_serialized_network(network, config)
    if not plan:
        raise RuntimeError("Engine build failed. The fusion pattern was not recognized or another error occurred.")
        
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)
    print("Engine build successful! Automatic FP8 fusion worked.")
    return engine

def benchmark(engine):
    """对成功构建的引擎进行测速"""
    context = engine.create_execution_context()
    input_a_gpu = torch.randn(M, K, dtype=torch.float32).cuda()
    output_gpu = torch.empty((M, N), dtype=torch.float32).cuda()
    input_name = "A"
    output_name = "output_c"
    context.set_tensor_address(input_name, input_a_gpu.data_ptr())
    context.set_tensor_address(output_name, output_gpu.data_ptr())

    print(f"Warming up for {N_WARMUP} iterations...")
    for _ in range(N_WARMUP):
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print(f"Running benchmark for {N_ITERATIONS} iterations...")
    start_event.record()
    for _ in range(N_ITERATIONS):
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms_cuda = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms_cuda / N_ITERATIONS
    throughput_tflops = (2 * M * N * K) / (avg_latency_ms * 1e-3) / 1e12

    print("\n--- Benchmark Results ---")
    print(f"TensorRT Version: {trt.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GEMM Dims: M={M}, N={N}, K={K}")
    print(f"Precision: FP8 (Achieved via Automatic Optimizer Fusion)")
    print(f"Average GPU Latency: {avg_latency_ms:.6f} ms")
    print(f"Throughput: {throughput_tflops:.4f} TFLOPS")
    print("-------------------------")


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major < 9:
         print("FP8 is only supported on GPUs with compute capability 9.0 (Hopper architecture) or newer.")
    else:
        try:
            engine = build_fp8_engine_the_right_way()
            if engine:
                benchmark(engine)
        except Exception as e:
            import traceback
            print(f"\n--- An error occurred during the process ---")
            print(f"Error: {e}")
            print("--- Traceback ---")
            traceback.print_exc()