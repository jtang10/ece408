
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define MAX_NUM_THREADS 1024
#define TILE_WIDTH  16
#define TILE_HEIGHT 16

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

#define getC(i) i/(K*K)
#define getK1(i) (i%(K*K))/K
#define getK2(i) (i%(K*K))%K

namespace mxnet 
{
namespace op 
{
__global__ void forward_shared_unroll(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out) {
  __shared__ float shmem_X[TILE_WIDTH][TILE_WIDTH*2];
  __shared__ float shmem_K[TILE_WIDTH][TILE_WIDTH];

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x * 2 + tx;
  int numMatACol = C * K * K;

  float acc = 0;
  float acc2 = 0;
  int temp_col = tx;
  int temp_row = ty;

  int X_h = col / W_out;
  int X_w = col % W_out;

  int X_h2 = (col + TILE_WIDTH) / W_out;
  int X_w2 = (col + TILE_WIDTH) % W_out;

  int K_c  = getC(temp_col);
  int K_k1 = getK1(temp_col);
  int K_k2 = getK2(temp_col);
  int X_c = getC(temp_row);
  int X_p = getK1(temp_row);
  int X_q = getK2(temp_row);
  float temp_k = k4d(row, K_c, K_k1, K_k2);
  float temp_x  = x4d(bz, X_c, X_h + X_p,  X_w + X_q);
  float temp_x2 = x4d(bz, X_c, X_h2 + X_p, X_w2 + X_q);

  #pragma unroll
  for (int i = 0; i < (numMatACol + TILE_WIDTH - 1) / (TILE_WIDTH); ++i) {
    if (temp_col < numMatACol && row < M) {
      shmem_K[ty][tx] = temp_k;
    } else {
      shmem_K[ty][tx] = 0;
    }

    if (temp_row < numMatACol && col < H_out * W_out) {
      shmem_X[ty][tx] = temp_x;
    } else {
      shmem_X[ty][tx] = 0;
    }

    if (temp_row < numMatACol && col + TILE_WIDTH < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH] = temp_x2;
    } else {
      shmem_X[ty][tx + TILE_WIDTH] = 0;
    }
    __syncthreads();

    temp_col += TILE_WIDTH;
    temp_row += TILE_WIDTH;
    K_c  = getC(temp_col);
    K_k1 = getK1(temp_col);
    K_k2 = getK2(temp_col);
    X_c = getC(temp_row);
    X_p = getK1(temp_row);
    X_q = getK2(temp_row);
    temp_k = k4d(row, K_c, K_k1, K_k2);
    temp_x  = x4d(bz, X_c, X_h + X_p,  X_w + X_q);
    temp_x2 = x4d(bz, X_c, X_h2 + X_p, X_w2 + X_q);

    #pragma unroll
    for (int q = 0; q < TILE_WIDTH; ++q) {
      acc  += shmem_K[ty][q] * shmem_X[q][tx];
      acc2 += shmem_K[ty][q] * shmem_X[q][tx + TILE_WIDTH];
    }

    __syncthreads();

    // int Y_b = b;
    // int Y_m = row;
    // int Y_h = col / W_out;
    // int Y_w = col % W_out;

    if (row < M && col < W_out * H_out) {
      y4d(bz, row, X_h, X_w) = acc;
    }

    if (row < M && col + TILE_WIDTH < W_out * H_out) {
      y4d(bz, row, X_h2, X_w2) = acc2;
    }
  }
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
  // printf("X: %u X %u X %u X %u matrix\n", x.size(0), x.size(1), x.size(2), x.size(3));
  // printf("K: %u X %u X %u X %u matrix\n", w.size(0), w.size(1), w.size(2), w.size(3));
  // printf("Y: %u X %u X %u X %u matrix\n", y.size(0), y.size(1), y.size(2), y.size(3));
  // X: 10000 X 1 X 72 X 72 matrix
  // K: 12 X 1 X 7 X 7 matrix
  // Y: 10000 X 12 X 66 X 66 matrix
  // Op Time: 0.071066
  // X: 10000 X 12 X 33 X 33 matrix
  // K: 24 X 12 X 7 X 7 matrix
  // Y: 10000 X 24 X 27 X 27 matrix
  // Op Time: 0.180310


  const int B = x.shape_[0];
  const int M = y.shape_[1];
  const int C = x.shape_[1];
  const int H = x.shape_[2];
  const int W = x.shape_[3];
  const int K = w.shape_[3];

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // newest kernel based on exam2
  dim3 gridDim(ceil(1.0*H_out*W_out/TILE_WIDTH), ceil(1.0*M/TILE_WIDTH), B);
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  forward_shared_unroll<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K, H_out, W_out);
  // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
  MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
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