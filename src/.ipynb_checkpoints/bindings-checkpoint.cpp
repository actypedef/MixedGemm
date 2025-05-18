#include <torch/extension.h>

// Include all files
#include "gemm.h"
#include "mxdtype.h"


torch::Tensor matmul(
        const torch::Tensor *AN,
        const torch::Tensor *BN,
        const torch::Tensor *AS,
        const torch::Tensor *BS,
        const torch::Tensor *AO,
        const torch::Tensor *BO,
        const torch::Tensor *SFAN,
        const torch::Tensor *SFBN,
        const torch::Tensor *SFAS,
        const torch::Tensor *SFBS,
        const torch::Tensor *SFAO,
        const torch::Tensor *SFBO
)
// {
//     torch::checkAllContiguous("matmul", {{A, "A",       0},
//                                                 {B, "B", 1}});
    torch::checkDeviceType("matmul", {AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO}, at::DeviceType::CUDA);

    // torch::checkAllSameGPU("matmul", {{A, "A",       0},
    //                                       {   B, "B", 1}});
    uint32_t M = AN.size(0);
    uint32_t N = BN.size(0);
    uint32_t KN = AN.size(1) * kElementsPerVectorFp4;  // 4bit packing is on the columns
    uint32_t KS = AS.size(1) * kElementsPerVectorFp6;  // 6bit packing is on the columns
    uint32_t KO = AO.size(1) * kElementsPerVectorFp8;  // 8bit packing is on the columns
    auto C = torch::zeros({M, N}, torch::dtype(torch::kBFloat16).device(AN.device()));
    // cutlass::NumericConverter<cutlass::float_ue8m0_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
    matmul_host(
        (cutlass::float_e2m1_t *)AN.data_ptr<Fp4Storage>(), (cutlass::float_e2m1_t *)BN.data_ptr<Fp4Storage>(),
        (cutlass::float_e3m2_t *)AS.data_ptr<Fp6Storage>(), (cutlass::float_e2m1_t *)BS.data_ptr<Fp4Storage>(), 
        (cutlass::float_e4m3_t *)AO.data_ptr<Fp8Storage>(), (cutlass::float_e2m1_t *)BO.data_ptr<Fp4Storage>(),
        M, N,
        KN, KS, KO,
        C.data_ptr<at::BFloat16>(), D.data_ptr<BFloat16>(),
        (cutlass::float_ue8m0_t *)SFAN.data_ptr<SF8Storage>(), (cutlass::float_ue8m0_t *)SFBN.data_ptr<SF8Storage>(),
        (cutlass::float_ue8m0_t *)SFAS.data_ptr<SF8Storage>(), (cutlass::float_ue8m0_t *)SFBS.data_ptr<SF8Storage>(),
        (cutlass::float_ue8m0_t *)SFAO.data_ptr<SF8Storage>(), (cutlass::float_ue8m0_t *)SFBO.data_ptr<SF8Storage>()
    );
    
    return C;
}

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{

    m.def("matmul", &matmul,
          "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
          "UINT8, CUDA))\n"
          "output: torch.Tensor(M x N, INT32, CUDA)\n"
          "output = int4Unpacking(A) @ int4Unpacking(B)^T",
          py::arg("A"), py::arg("B"));

}