// Fused ReLU + BatchNorm kernel (inference mode)
// Combines two operations into one CUDA kernel to reduce memory I/O

#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_relu_bn_kernel(
    const float* __restrict__ input,     // (N, C, H, W)
    const float* __restrict__ gamma,     // (C,) - scale
    const float* __restrict__ beta,      // (C,) - offset
    float* __restrict__ output,          // (N, C, H, W)
    int N, int C, int H, int W,
    float epsilon = 1e-5f) {

    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;

    if (idx >= total_elements) return;

    // Compute indices
    int spatial_idx = idx % (H * W);
    int channel = (idx / (H * W)) % C;
    int batch = idx / (C * H * W);

    // Load input value
    float x = input[idx];

    // Load per-channel batch norm parameters
    float scale = gamma[channel];
    float offset = beta[channel];

    // Apply ReLU first (before BN in training, but combined in inference)
    // In inference mode, BN computes: y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
    // For this fused version, we assume x is already normalized (or approximated)
    // This is a simplified version - real batch norm would use running statistics

    float normalized = x;  // In real implementation: (x - mean) / sqrt(var + eps)

    // Apply scale and shift
    float bn_output = normalized * scale + offset;

    // Apply ReLU
    float result = fmaxf(bn_output, 0.0f);

    // Store result
    output[idx] = result;
}

// Wrapper for PyTorch
extern "C" {
    cudaError_t launch_fused_relu_bn(
        const float* input,
        const float* gamma,
        const float* beta,
        float* output,
        int N, int C, int H, int W,
        cudaStream_t stream = 0) {

        int total_elements = N * C * H * W;
        int threads_per_block = 256;
        int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        fused_relu_bn_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            input, gamma, beta, output, N, C, H, W);

        return cudaGetLastError();
    }
}
