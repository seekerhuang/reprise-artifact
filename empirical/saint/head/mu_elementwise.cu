 
#include <cuda_runtime.h>
#include <cstdint>

__global__ void product_kernel(const float* boundary, const float* lambda,
                               double* product, std::int64_t count) {
  for (std::int64_t i=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       i<count; i+=static_cast<std::int64_t>(blockDim.x)*gridDim.x) {
    const double b=static_cast<double>(boundary[i]);
    const double l=static_cast<double>(lambda[i]);
    product[i]=__dmul_rn(b,l);
  }
}

__global__ void residual_kernel(const float* z, const double* product,
                                const double* row_inner, const bool* branch,
                                double* residual, std::int64_t count) {
  for (std::int64_t i=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
       i<count; i+=static_cast<std::int64_t>(blockDim.x)*gridDim.x) {
    const std::int64_t row=i/512;
    const double zd=static_cast<double>(z[i]);
    const double square=__dmul_rn(zd,zd);
    const double scaled=__dmul_rn(square,row_inner[row]);
    const double selected=__dmul_rn(scaled,static_cast<double>(branch[row]));
    residual[i]=__dsub_rn(product[i],selected);
  }
}

static int validate(const void* a,const void* b,const void* out,
                    std::int64_t count,int expected_device) {
  if (!a || !b || !out || count<=0 || count%512) return cudaErrorInvalidValue;
  int actual=-1;
  const cudaError_t code=cudaGetDevice(&actual);
  if (code!=cudaSuccess) return code;
  if (actual!=expected_device) return cudaErrorInvalidDevice;
  return cudaSuccess;
}

extern "C" int mu_elementwise_abi_version() { return 1; }

extern "C" int mu_product(const float* boundary,const float* lambda,double* product,
                           std::int64_t count,void* stream,int device) {
  const int code=validate(boundary,lambda,product,count,device);
  if (code) return code;
  const std::int64_t needed=(count+255)/256;
  const unsigned blocks=static_cast<unsigned>(needed<65535 ? needed : 65535);
  product_kernel<<<blocks,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(boundary,lambda,product,count);
  return cudaGetLastError();
}

extern "C" int mu_residual(const float* z,const double* product,const double* row_inner,
                            const bool* branch,double* residual,std::int64_t count,
                            void* stream,int device) {
  const int code=validate(z,product,residual,count,device);
  if (code) return code;
  if (!row_inner || !branch) return cudaErrorInvalidValue;
  const std::int64_t needed=(count+255)/256;
  const unsigned blocks=static_cast<unsigned>(needed<65535 ? needed : 65535);
  residual_kernel<<<blocks,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(z,product,row_inner,branch,residual,count);
  return cudaGetLastError();
}
