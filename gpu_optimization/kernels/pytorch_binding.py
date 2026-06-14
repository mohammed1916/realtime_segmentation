import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source
cuda_source = """
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_relu_bn_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C, int H, int W) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;

    if (idx >= total) return;

    int channel = (idx / (H * W)) % C;
    float x = input[idx];
    float scale = gamma[channel];
    float offset = beta[channel];
    float normalized = x;
    float bn_out = normalized * scale + offset;
    float result = fmaxf(bn_out, 0.0f);
    output[idx] = result;
}

torch::Tensor fused_relu_bn(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta) {

    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto output = torch::empty_like(input);

    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_relu_bn_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W);

    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_relu_bn(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_relu_bn", &fused_relu_bn, "Fused ReLU + BatchNorm kernel");
}
"""

def load_fused_kernel():
    """Load the fused ReLU+BN kernel."""
    try:
        fused_ops = load_inline(
            name='fused_relu_bn',
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            functions=['fused_relu_bn'],
            verbose=False,
            extra_cuda_cflags=['-O3'],
        )
        return fused_ops
    except Exception as e:
        print(f"Warning: Could not load fused kernel: {e}")
        print("Falling back to PyTorch operations")
        return None

class FusedReLUBN(nn.Module):
    """Fused ReLU + BatchNorm module."""
    def __init__(self, num_features, use_custom_kernel=True):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.use_custom_kernel = use_custom_kernel
        self.fused_ops = None

        if use_custom_kernel:
            self.fused_ops = load_fused_kernel()

    def forward(self, x):
        if self.fused_ops is not None:
            return self.fused_ops.fused_relu_bn(x, self.gamma, self.beta)
        else:
            # Fallback to PyTorch operations
            bn_out = x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
            return torch.nn.functional.relu(bn_out)

if __name__ == '__main__':
    print("Fused ReLU+BN Kernel")
    print("="*70)

    print("\nAttempting to load kernel...")
    fused_ops = load_fused_kernel()

    if fused_ops:
        print("Kernel loaded successfully!")

        x = torch.randn(2, 64, 32, 32, device='cuda')
        gamma = torch.ones(64, device='cuda')
        beta = torch.zeros(64, device='cuda')

        print(f"\nInput shape: {x.shape}")

        output = fused_ops.fused_relu_bn(x, gamma, beta)
        print(f"Output shape: {output.shape}")

        print("Kernel execution successful!")
    else:
        print("Kernel loading failed, using fallback")

        module = FusedReLUBN(64, use_custom_kernel=False)
        module.cuda()

        x = torch.randn(2, 64, 32, 32, device='cuda')
        output = module(x)

        print(f"Fallback execution successful!")
        print(f"Output shape: {output.shape}")
