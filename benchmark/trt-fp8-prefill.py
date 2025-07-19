import tensorrt as trt
import torch
import numpy as np
import time
import traceback

# 1. 定义 Llama 3-8B 完整模型的结构参数
BATCH_SIZE = 1
MAX_SEQ_LEN = 4096
HIDDEN_SIZE = 4096
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
FFN_HIDDEN_SIZE = 11008
GQA_FACTOR = NUM_ATTENTION_HEADS // NUM_KV_HEADS
VOCAB_SIZE = 128256 # Llama 3 的词汇表大小
NUM_LAYERS = 32      # <--- 新增：Llama 3-8B 的层数

# TensorRT 日志记录器
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def create_dummy_weights(shape, dtype=np.float16):
    """创建一个包含随机数据的 trt.Weights 对象"""
    return trt.Weights(np.random.rand(*shape).astype(dtype))

def _create_rope_cache(network, dtype):
    """预计算 RoPE 的 sin 和 cos 缓存并作为常量添加到网络中"""
    print("Creating RoPE cache for MAX_SEQ_LEN...")
    theta_base = 10000.0
    
    theta = theta_base ** (-2.0 * np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM)
    position = np.arange(MAX_SEQ_LEN, dtype=np.float32)
    freqs = np.outer(position, theta)
    
    emb = np.concatenate((freqs, freqs), axis=-1)
    
    cos_cache = np.cos(emb)[np.newaxis, :, np.newaxis, :].astype(dtype)
    sin_cache = np.sin(emb)[np.newaxis, :, np.newaxis, :].astype(dtype)
    
    cos_cache_reshaped = np.transpose(cos_cache, (0, 2, 1, 3))
    sin_cache_reshaped = np.transpose(sin_cache, (0, 2, 1, 3))

    cos_const = network.add_constant(cos_cache_reshaped.shape, trt.Weights(cos_cache_reshaped)).get_output(0)
    sin_const = network.add_constant(sin_cache_reshaped.shape, trt.Weights(sin_cache_reshaped)).get_output(0)
    
    return cos_const, sin_const

def add_rope(network, input_tensor, cos_cache, sin_cache):
    """在网络中添加 RoPE 层 (修正版v2，统一数据类型)"""
    head_dim_half = HEAD_DIM // 2

    # 1. 获取输入张量的动态形状。这将创建一个 1D 张量，通常是 Int64 类型。
    shape_of_input = network.add_shape(input_tensor).get_output(0)

    # 2. 从动态形状张量中提取前三个维度 [B, H, S]。
    slice_for_bhs = network.add_slice(shape_of_input, start=[0], shape=[3], stride=[1]).get_output(0)
    
    # --- 核心修正：将常量的数据类型与 shape 张量统一 ---
    # 3. 创建一个包含 D/2 的常量张量，使用 np.int64 来匹配 add_shape 的输出类型。
    const_head_dim_half = network.add_constant((1,), trt.Weights(np.array([head_dim_half], dtype=np.int64))).get_output(0)

    # 4. 将它们连接起来，形成最终的动态切片形状 [B, H, S, D/2]。现在类型匹配了。
    dynamic_slice_shape = network.add_concatenation([slice_for_bhs, const_head_dim_half]).get_output(0)
    
    # 5. 使用 add_slice，通过 set_input(2, ...) 将动态形状作为输入。
    slice_layer1 = network.add_slice(input_tensor, start=[0,0,0,0], shape=[], stride=[1,1,1,1])
    slice_layer1.set_input(2, dynamic_slice_shape)
    
    slice_layer2 = network.add_slice(input_tensor, start=[0,0,0,head_dim_half], shape=[], stride=[1,1,1,1])
    slice_layer2.set_input(2, dynamic_slice_shape)

    x1 = slice_layer1.get_output(0)
    x2 = slice_layer2.get_output(0)

    neg_x2 = network.add_unary(x2, trt.UnaryOperation.NEG).get_output(0)

    concat_layer = network.add_concatenation([neg_x2, x1])
    concat_layer.axis = 3
    rotated_input = concat_layer.get_output(0)

    cos_term = network.add_elementwise(input_tensor, cos_cache, trt.ElementWiseOperation.PROD).get_output(0)
    sin_term = network.add_elementwise(rotated_input, sin_cache, trt.ElementWiseOperation.PROD).get_output(0)
    
    output_tensor = network.add_elementwise(cos_term, sin_term, trt.ElementWiseOperation.SUM).get_output(0)
    return output_tensor

def add_rmsnorm(network, input_tensor, weight_shape, op_name=""):
    """在网络中添加 RMSNorm 层"""
    epsilon = 1e-5
    dtype = np.float16

    pow2_tensor = network.add_elementwise(input_tensor, input_tensor, trt.ElementWiseOperation.PROD).get_output(0)
    reduce_axes = 1 << (len(input_tensor.shape) - 1)
    mean_tensor = network.add_reduce(pow2_tensor, trt.ReduceOperation.AVG, axes=reduce_axes, keep_dims=True).get_output(0)
    epsilon_tensor = network.add_constant(shape=(1,) * len(input_tensor.shape), weights=trt.Weights(np.array([epsilon], dtype=dtype))).get_output(0)
    add_eps_tensor = network.add_elementwise(mean_tensor, epsilon_tensor, trt.ElementWiseOperation.SUM).get_output(0)
    sqrt_tensor = network.add_unary(add_eps_tensor, trt.UnaryOperation.SQRT).get_output(0)
    reciprocal_sqrt_tensor = network.add_unary(sqrt_tensor, trt.UnaryOperation.RECIP).get_output(0)
    normalized_tensor = network.add_elementwise(input_tensor, reciprocal_sqrt_tensor, trt.ElementWiseOperation.PROD).get_output(0)
    
    weight_const_1d = network.add_constant(weight_shape, create_dummy_weights(weight_shape, dtype=dtype)).get_output(0)
    weight_const_1d.name = f"{op_name}_weight"
    
    shuffle_layer = network.add_shuffle(weight_const_1d)
    reshape_dims = [1] * len(input_tensor.shape); reshape_dims[-1] = weight_shape[0]
    shuffle_layer.reshape_dims = tuple(reshape_dims)
    weight_const_reshaped = shuffle_layer.get_output(0)
    
    return network.add_elementwise(normalized_tensor, weight_const_reshaped, trt.ElementWiseOperation.PROD).get_output(0)


def build_llama3_engine():
    """构建使用 FP8 的 Llama 3 完整模型 TensorRT 引擎"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()

    profile = builder.create_optimization_profile()
    # --- 修改：输入现在是 token_ids，形状为 [batch, seq_len] ---
    profile.set_shape("input_token_ids", (BATCH_SIZE, 1), (BATCH_SIZE, 2048), (BATCH_SIZE, MAX_SEQ_LEN))
    config.add_optimization_profile(profile)

    scale_val = 0.1
    scale_tensor = network.add_constant((), trt.Weights(np.array([scale_val], dtype=np.float32))).get_output(0)
    scale_tensor.name = "Global_FP8_Scale"

    def add_fp8_linear_op(input_tensor, weight_tensor, op_type, equation=None, transpose_b=False, op_name=""):
        input_q = network.add_quantize(input_tensor, scale_tensor, trt.DataType.FP8); input_q.name = f"{op_name}_input_quant"
        input_dq = network.add_dequantize(input_q.get_output(0), scale_tensor, trt.float16); input_dq.name = f"{op_name}_input_dequant"
        dequantized_input = input_dq.get_output(0)
        weight_q = network.add_quantize(weight_tensor, scale_tensor, trt.DataType.FP8); weight_q.name = f"{op_name}_weight_quant"
        weight_dq = network.add_dequantize(weight_q.get_output(0), scale_tensor, trt.float16); weight_dq.name = f"{op_name}_weight_dequant"
        dequantized_weight = weight_dq.get_output(0)
        if op_type == 'einsum': layer = network.add_einsum([dequantized_input, dequantized_weight], equation)
        elif op_type == 'matmul': layer = network.add_matrix_multiply(dequantized_input, trt.MatrixOperation.NONE, dequantized_weight, trt.MatrixOperation.TRANSPOSE if transpose_b else trt.MatrixOperation.NONE)
        else: raise ValueError(f"Unsupported op_type: {op_type}")
        layer.name = op_name
        return layer.get_output(0)
    
    # --- 新增：模型输入层 ---
    input_ids = network.add_input(name="input_token_ids", dtype=trt.int32, shape=(BATCH_SIZE, -1))
    
    # --- 新增：词嵌入层 (Embedding Layer) ---
    embedding_weights = network.add_constant((VOCAB_SIZE, HIDDEN_SIZE), create_dummy_weights((VOCAB_SIZE, HIDDEN_SIZE))).get_output(0)
    embedding_layer = network.add_gather(embedding_weights, input_ids, axis=0)
    hidden_states = embedding_layer.get_output(0)

    # 预先创建所有解码器层都会用到的 RoPE 缓存
    cos_cache, sin_cache = _create_rope_cache(network, np.float16)

    # --- 核心修改：循环构建 32 个解码器层 ---
    for i in range(NUM_LAYERS):
        print(f"Building Decoder Layer {i+1}/{NUM_LAYERS}...")
        
        # 每次循环都使用上一层的输出作为输入
        layer_input = hidden_states 
        
        # --- 1. 注意力块 ---
        # 输入归一化
        normed_input = add_rmsnorm(network, layer_input, (HIDDEN_SIZE,), op_name=f"layer_{i}_attn_norm")
        
        # 为当前层创建唯一的权重
        q_proj_w = network.add_constant((HIDDEN_SIZE, HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, HIDDEN_SIZE))).get_output(0)
        k_proj_w = network.add_constant((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM), create_dummy_weights((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM))).get_output(0)
        v_proj_w = network.add_constant((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM), create_dummy_weights((HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM))).get_output(0)
        o_proj_w = network.add_constant((HIDDEN_SIZE, HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, HIDDEN_SIZE))).get_output(0)

        q_proj = add_fp8_linear_op(normed_input, q_proj_w, 'einsum', "bsk,kn->bsn", op_name=f"layer_{i}_Q_Proj")
        k_proj = add_fp8_linear_op(normed_input, k_proj_w, 'einsum', "bsk,kn->bsn", op_name=f"layer_{i}_K_Proj")
        v_proj = add_fp8_linear_op(normed_input, v_proj_w, 'einsum', "bsk,kn->bsn", op_name=f"layer_{i}_V_Proj")

        # Reshape & RoPE
        q_shuffle = network.add_shuffle(q_proj); q_shuffle.reshape_dims = (BATCH_SIZE, -1, NUM_ATTENTION_HEADS, HEAD_DIM); q_shuffle.second_transpose = trt.Permutation([0, 2, 1, 3])
        k_shuffle = network.add_shuffle(k_proj); k_shuffle.reshape_dims = (BATCH_SIZE, -1, NUM_KV_HEADS, HEAD_DIM); k_shuffle.second_transpose = trt.Permutation([0, 2, 1, 3])
        v_shuffle = network.add_shuffle(v_proj); v_shuffle.reshape_dims = (BATCH_SIZE, -1, NUM_KV_HEADS, HEAD_DIM); v_shuffle.second_transpose = trt.Permutation([0, 2, 1, 3])
        
        q_with_rope = add_rope(network, q_shuffle.get_output(0), cos_cache, sin_cache)
        k_with_rope = add_rope(network, k_shuffle.get_output(0), cos_cache, sin_cache)
        
        # GQA
        k_repeated = network.add_concatenation([k_with_rope] * GQA_FACTOR).get_output(0) if GQA_FACTOR > 1 else k_with_rope
        v_repeated = network.add_concatenation([v_shuffle.get_output(0)] * GQA_FACTOR).get_output(0) if GQA_FACTOR > 1 else v_shuffle.get_output(0)

        # Attention 计算
        qkT = add_fp8_linear_op(q_with_rope, k_repeated, 'matmul', transpose_b=True, op_name=f"layer_{i}_QKT_MatMul")
        scale_const = network.add_constant((1,1,1,1), trt.Weights(np.array([1.0 / (HEAD_DIM ** 0.5)], dtype=np.float16))).get_output(0)
        qkT_scaled = network.add_elementwise(qkT, scale_const, trt.ElementWiseOperation.PROD).get_output(0)
        softmax_layer = network.add_softmax(qkT_scaled); softmax_layer.axes = 1 << 3
        attention_probs = softmax_layer.get_output(0)
        attn_out_bshd = add_fp8_linear_op(attention_probs, v_repeated, 'matmul', op_name=f"layer_{i}_ProbsV_MatMul")

        # Reshape back
        shuffle_out = network.add_shuffle(attn_out_bshd); shuffle_out.first_transpose = trt.Permutation([0, 2, 1, 3]); shuffle_out.reshape_dims = (BATCH_SIZE, -1, HIDDEN_SIZE)
        
        attention_output = add_fp8_linear_op(shuffle_out.get_output(0), o_proj_w, 'einsum', "bsk,kn->bsn", op_name=f"layer_{i}_O_Proj")
        
        # 第一个残差连接
        residual1 = network.add_elementwise(layer_input, attention_output, trt.ElementWiseOperation.SUM).get_output(0)

        # --- 2. FFN 块 ---
        normed_residual1 = add_rmsnorm(network, residual1, (HIDDEN_SIZE,), op_name=f"layer_{i}_ffn_norm")
        
        ffn_gate_w = network.add_constant((HIDDEN_SIZE, FFN_HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, FFN_HIDDEN_SIZE))).get_output(0)
        ffn_up_w = network.add_constant((HIDDEN_SIZE, FFN_HIDDEN_SIZE), create_dummy_weights((HIDDEN_SIZE, FFN_HIDDEN_SIZE))).get_output(0)
        ffn_down_w = network.add_constant((FFN_HIDDEN_SIZE, HIDDEN_SIZE), create_dummy_weights((FFN_HIDDEN_SIZE, HIDDEN_SIZE))).get_output(0)

        gate_proj = add_fp8_linear_op(normed_residual1, ffn_gate_w, 'einsum', "bsk,kn->bsn", op_name=f"layer_{i}_FFN_Gate")
        up_proj = add_fp8_linear_op(normed_residual1, ffn_up_w, 'einsum', "bsk,kn->bsn", op_name=f"layer_{i}_FFN_Up")
        
        sigmoid_gate = network.add_activation(gate_proj, trt.ActivationType.SIGMOID).get_output(0)
        silu_out = network.add_elementwise(gate_proj, sigmoid_gate, trt.ElementWiseOperation.PROD).get_output(0)
        gated_ffn = network.add_elementwise(silu_out, up_proj, trt.ElementWiseOperation.PROD).get_output(0)
        
        ffn_output = add_fp8_linear_op(gated_ffn, ffn_down_w, 'einsum', "bsk,kn->bsn", op_name=f"layer_{i}_FFN_Down")
        
        # 第二个残差连接，其输出成为下一层的输入
        hidden_states = network.add_elementwise(residual1, ffn_output, trt.ElementWiseOperation.SUM).get_output(0)
        hidden_states.name = f"layer_{i}_output"

    # --- 新增：模型输出层 ---
    print("Building Final Layers (Norm and LM Head)...")
    
    # 最终归一化
    final_norm_output = add_rmsnorm(network, hidden_states, (HIDDEN_SIZE,), op_name="final_norm")
    
    # LM Head: 线性层将 hidden_states 映射到词汇表 logits
    lm_head_weights = network.add_constant((HIDDEN_SIZE, VOCAB_SIZE), create_dummy_weights((HIDDEN_SIZE, VOCAB_SIZE))).get_output(0)
    # 对于 LM head，通常使用 FP16 以保证精度，所以这里不使用 FP8 辅助函数
    logits_matmul_layer = network.add_matrix_multiply(final_norm_output, trt.MatrixOperation.NONE, lm_head_weights, trt.MatrixOperation.NONE)
    logits = logits_matmul_layer.get_output(0)
    
    # --- 修改：标记最终输出 ---
    logits.name = "output_logits"
    network.mark_output(logits)
    
    print("\nBuilding Full Model TensorRT engine... (This will take several minutes and consume significant memory)")
    plan = builder.build_serialized_network(network, config)
    if not plan:
        print("ERROR: Engine build failed. This could be due to memory constraints or an issue in the network definition.")
        return None
    
    print("Engine build successful!")
    return plan

def benchmark(engine_plan):
    """使用构建的完整模型引擎对不同的 prefill 序列长度进行性能测试"""
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_plan)
    context = engine.create_execution_context()

    prefill_lengths = [128, 512, 1024, 2048, 4096]
    
    print("\n--- Full Model Prefill Benchmark Results ---")
    print(f"Batch Size: {BATCH_SIZE}, Num Layers: {NUM_LAYERS}")
    print(f"Precision: FP8 (triggered by fusible Q/Dq pattern)")
    print("--------------------------------------------------")

    for seq_len in prefill_lengths:
        if seq_len > MAX_SEQ_LEN:
            print(f"Skipping seq_len {seq_len} as it exceeds MAX_SEQ_LEN {MAX_SEQ_LEN}")
            continue

        print(f"\nBenchmarking for Sequence Length: {seq_len}...")
        
        input_shape = (BATCH_SIZE, seq_len)
        context.set_input_shape("input_token_ids", input_shape)
        output_shape = context.get_tensor_shape("output_logits")
        
        input_tensor = torch.randint(0, VOCAB_SIZE, input_shape, dtype=torch.int32).cuda()
        output_tensor = torch.empty(output_shape, dtype=torch.float16).cuda()
        
        context.set_tensor_address("input_token_ids", input_tensor.data_ptr())
        context.set_tensor_address("output_logits", output_tensor.data_ptr())

        print("Warming up...")
        for _ in range(10):
            context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        num_runs = max(10, 20480 // seq_len)
        print(f"Running benchmark for {num_runs} iterations...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_runs):
            context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        end_event.record()
        torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency = total_time_ms / num_runs
        throughput = (BATCH_SIZE * seq_len) / (avg_latency / 1000) if avg_latency > 0 else 0

        print(f"  Average Latency: {avg_latency:.3f} ms")
        print(f"  Tokens per Second (Throughput): {throughput:.2f} tokens/sec")

    print("\n--------------------------------------------------")

if __name__ == "__main__":
    print(f"TensorRT version: {trt.__version__}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
    elif torch.cuda.get_device_properties(0).major < 9:
        print("ERROR: FP8 is only supported on GPUs with compute capability 9.0 (Hopper architecture) or newer.")
        print(f"Your GPU's compute capability is {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}. Aborting.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)} (Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor})")
        engine_plan = None
        try:
            # 修改函数调用
            engine_plan = build_llama3_engine()
            if engine_plan:
                benchmark(engine_plan)
        except Exception as e:
            print(f"\n--- An error occurred during the process ---")
            print(f"Error: {e}")
            print("--- Traceback ---")
            traceback.print_exc()