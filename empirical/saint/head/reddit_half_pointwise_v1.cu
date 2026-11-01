 
#include <cuda_runtime.h>
#include <cstdint>

__global__ void half_product_kernel(const float* boundary, const float* lambda0,
                                   const float* lambda1, double* product,
                                   std::int64_t count) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const double difference = __dsub_rn(static_cast<double>(lambda1[i]),
                                      static_cast<double>(lambda0[i]));
    const double half = __dmul_rn(difference, .5);
    product[i] = __dmul_rn(static_cast<double>(boundary[i]), half);
  }
}

__global__ void residual_kernel(const float* z, const double* product,
                                const double* row_inner, const bool* branch,
                                double* residual, std::int64_t count) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       i < count; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t row = i / 128;
    const double zd = static_cast<double>(z[i]);
    const double square = __dmul_rn(zd, zd);
    const double scaled = __dmul_rn(square, row_inner[row]);
    const double selected = __dmul_rn(scaled, static_cast<double>(branch[row]));
    residual[i] = __dsub_rn(product[i], selected);
  }
}

static int validate(const void* a, const void* b, const void* out,
                    std::int64_t count, int expected_device) {
  if (!a || !b || !out || count <= 0 || count % 128) return cudaErrorInvalidValue;
  int actual = -1;
  const cudaError_t code = cudaGetDevice(&actual);
  if (code != cudaSuccess) return code;
  if (actual != expected_device) return cudaErrorInvalidDevice;
  return cudaSuccess;
}

extern "C" int reddit_half_pointwise_abi_version() { return 1; }

extern "C" int reddit_half_product(const float* boundary, const float* lambda0,
    const float* lambda1, double* product, std::int64_t count, void* stream, int device) {
  const int code = validate(boundary, lambda0, product, count, device);
  if (code) return code;
  if (!lambda1) return cudaErrorInvalidValue;
  const auto needed = (count + 255) / 256;
  const unsigned blocks = static_cast<unsigned>(needed < 65535 ? needed : 65535);
  half_product_kernel<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      boundary, lambda0, lambda1, product, count);
  return cudaGetLastError();
}

extern "C" int reddit_half_residual(const float* z, const double* product,
    const double* row_inner, const bool* branch, double* residual,
    std::int64_t count, void* stream, int device) {
  const int code = validate(z, product, residual, count, device);
  if (code) return code;
  if (!row_inner || !branch) return cudaErrorInvalidValue;
  const auto needed = (count + 255) / 256;
  const unsigned blocks = static_cast<unsigned>(needed < 65535 ? needed : 65535);
  residual_kernel<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      z, product, row_inner, branch, residual, count);
  return cudaGetLastError();
}
