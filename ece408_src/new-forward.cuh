
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define MAX_NUM_THREADS 1024
#define TILE_WIDTH 32

namespace mxnet 
{
namespace op 
{
// im2col based on single image with dimension of [C, H, W]
// Output matrix has the dimension of [C, H_col, W_col] = [C, K*K, H_out*W_out]
// Each thread is aligned to
__global__ void unrollKernel(const float *x, float *x_col, const int C, const int H, const int W, const int K) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const int H_col = K * K;
  const int W_col = H_out * W_out;

  int c, s, p, q, h_out, w_out;
  if (index < C * W_col) {
    c = index / W_col;
    s = index % W_col;
    h_out = s / W_out;
    w_out = s % W_out;

    for (p = 0; p < K; ++p) {
      for (q = 0; q < K; ++q) {
        x_col[c*H_col*W_col + (p*K + q)*W_col + s] = x[c * H * W + (h_out + p) * W + (w_out + q)];
      }
    }
  }
}

__global__ void matrixMultiplykernel(float* k, float* x, float* y, int M, int H, int W)
{
  __shared__ float sharedW[TILE_WIDTH][TILE_WIDTH];
  __shared__ float sharedX[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;

  float Pvalue = 0;
  for (int m = 0; m < (H - 1) / TILE_WIDTH + 1; m++){
    if(m * TILE_WIDTH + tx < H && Row < M)
        sharedW[ty][tx] = k[Row * H + m * TILE_WIDTH + tx];
    else
      sharedW[ty][tx] = 0.0f;

    if(m * TILE_WIDTH + ty < H && Col < W)
        sharedX[ty][tx] = x[(m * TILE_WIDTH + ty) * W + Col];
    else
      sharedX[ty][tx] = 0.0f;

      __syncthreads();

      for(int i = 0; i < TILE_WIDTH; ++i) {
        Pvalue += sharedW[ty][i] * sharedX[i][tx];
      }

      __syncthreads();
  }

  if(Row < M && Col < W){
    y[Row * W + Col] = Pvalue;
  }
}

__global__ void forward_kernel(float *y, const float *x, const float *k, 
                               const int B, const int M, const int C, 
                               const int H, const int W, const int K) {

  /*
  Modify this function to implement the forward pass described in Chapter 16.
  We have added an additional dimension to the tensors to support an entire mini-batch
  The goal here is to be correct AND fast.
  We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.

  x: B C H W
  y: B M H W
  k: M C K K
  */

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // An example use of these macros:
  // float a = y4d(0,0,0,0)
  // y4d(0,0,0,0) = a
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  int bx = blockIdx.x;  int by = blockIdx.y;  // int bz = blockIdx.z;
  int tx = threadIdx.x; int ty = threadIdx.y; // int tz = threadIdx.z;
  
  const int idx_x = blockDim.x * bx + tx;
  const int idx_y = blockDim.y * by + ty;

  const int b = idx_y / H_out;
  const int m = idx_x / W_out;
  const int h = idx_y % H_out;
  const int w = idx_x % W_out;

  if (idx_x < W_out * M && idx_y < H_out * B) {
    float acc = 0;
    for(int c = 0; c < C; ++c) {
      for(int p = 0; p < K; ++p) {
        for(int q = 0; q < K; ++q) {
          acc += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
        }
      }
    }
    y4d(b, m, h, w) = acc;
  }

  #undef y4d
  #undef x4d
  #undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called,
   so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y,
                         const mshadow::Tensor<gpu, 4, float> &x, 
                         const mshadow::Tensor<gpu, 4, float> &w) {
  // Extract the tensor dimensions into B,M,C,H,W,K
  printf("%u X %u X %u X %u matrix\n", x.size(0), x.size(1), x.size(2), x.size(3));

  const int B = x.shape_[0];
  const int M = y.shape_[1];
  const int C = x.shape_[1];
  const int H = x.shape_[2];
  const int W = x.shape_[3];
  const int K = w.shape_[3];

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // Working on unrolling x
  float *x_unrolled;
  
  const int H_unroll = C*K*K;
  const int W_unroll = W_out * H_out;
  cudaMalloc((void**) &x_unrolled, H_unroll * W_unroll * sizeof(float));

  // Set the kernel dimensions
  dim3 gridDim_unroll(ceil(1.0 * C * H_out * W_out / MAX_NUM_THREADS));
  dim3 blockDim_unroll(MAX_NUM_THREADS);

  dim3 grid_Multiple(ceil(H_out*W_out/(TILE_WIDTH*1.0)), (M - 1)/TILE_WIDTH + 1, 1);
  dim3 block_Multiple(TILE_WIDTH, TILE_WIDTH, 1);

  for (int i = 0; i < B; ++i) {
    unrollKernel<<<gridDim_unroll, blockDim_unroll>>>(x.dptr_ + C*H*W*i, x_unrolled, C, H, W, K);
    matrixMultiplykernel<<<grid_Multiple, block_Multiple>>>(w.dptr_, x_unrolled, y.dptr_ + M*H_out*W_out*i,  
                                                            M, H_unroll, W_unroll);
  }




  // Call the kernel
  // forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

  // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
  MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
  cudaFree(x_unrolled);
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y,
             const mshadow::Tensor<gpu, 4, DType> &x,
             const mshadow::Tensor<gpu, 4, DType> &w) {
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}

}
}

#endif