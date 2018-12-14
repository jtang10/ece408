
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include "cuda_fp16.h"

#define MAX_NUM_THREADS 1024
#define TILE_WIDTH_ONE  12
#define TILE_WIDTH_TWO  24
// #define TILE_HEIGHT 16
#define GRANULARITY 8

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
__global__ void 
__launch_bounds__(TILE_WIDTH_ONE * TILE_WIDTH_ONE, 4)
forward_shared_unroll_one(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out) {
  __shared__ half2 shmem_X[TILE_WIDTH_ONE][TILE_WIDTH_ONE * GRANULARITY / 2];
  __shared__ half2 shmem_K[TILE_WIDTH_ONE][TILE_WIDTH_ONE];

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x * GRANULARITY + tx;
  int numMatACol = C * K * K;

  half2 acc1 = __float2half2_rn(0.f);
  half2 acc2 = __float2half2_rn(0.f);
  half2 acc3 = __float2half2_rn(0.f);
  half2 acc4 = __float2half2_rn(0.f);

  int temp_col = tx;
  int temp_row = ty;

  int X_h1 = col / W_out;
  int X_w1 = col % W_out;

  int X_h2 = (col + TILE_WIDTH_ONE*1) / W_out;
  int X_w2 = (col + TILE_WIDTH_ONE*1) % W_out;

  int X_h3 = (col + TILE_WIDTH_ONE*2) / W_out;
  int X_w3 = (col + TILE_WIDTH_ONE*2) % W_out;

  int X_h4 = (col + TILE_WIDTH_ONE*3) / W_out;
  int X_w4 = (col + TILE_WIDTH_ONE*3) % W_out;

  int X_h5 = (col + TILE_WIDTH_ONE*4) / W_out;
  int X_w5 = (col + TILE_WIDTH_ONE*4) % W_out;

  int X_h6 = (col + TILE_WIDTH_ONE*5) / W_out;
  int X_w6 = (col + TILE_WIDTH_ONE*5) % W_out;

  int X_h7 = (col + TILE_WIDTH_ONE*6) / W_out;
  int X_w7 = (col + TILE_WIDTH_ONE*6) % W_out;

  int X_h8 = (col + TILE_WIDTH_ONE*7) / W_out;
  int X_w8 = (col + TILE_WIDTH_ONE*7) % W_out;

  int X_c  = getC(temp_row);
  int X_p  = getK1(temp_row);
  int X_q  = getK2(temp_row);

  half temp_k  = __float2half_rn(k[row * numMatACol + temp_col]);
  float temp_x1 = x4d(bz, X_c, X_h1 + X_p, X_w1 + X_q);
  float temp_x2 = x4d(bz, X_c, X_h2 + X_p, X_w2 + X_q);
  float temp_x3 = x4d(bz, X_c, X_h3 + X_p, X_w3 + X_q);
  float temp_x4 = x4d(bz, X_c, X_h4 + X_p, X_w4 + X_q);
  float temp_x5 = x4d(bz, X_c, X_h5 + X_p, X_w5 + X_q);
  float temp_x6 = x4d(bz, X_c, X_h6 + X_p, X_w6 + X_q);
  float temp_x7 = x4d(bz, X_c, X_h7 + X_p, X_w7 + X_q);
  float temp_x8 = x4d(bz, X_c, X_h8 + X_p, X_w8 + X_q);

  #pragma unroll
  for (int i = 0; i < (numMatACol + TILE_WIDTH_ONE - 1) / (TILE_WIDTH_ONE); ++i) {
    if (temp_col < numMatACol && row < M) {
      shmem_K[ty][tx] = __half2half2(temp_k);
    } else {
      shmem_K[ty][tx] = __float2half2_rn(0.f);
    }

    if (temp_row < numMatACol && col < H_out * W_out) {
      shmem_X[ty][tx].x = __float2half_ru(temp_x1);
    } else {
      shmem_X[ty][tx].x = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_ONE < H_out * W_out) {
      shmem_X[ty][tx].y = __float2half_ru(temp_x2);
    } else {
      shmem_X[ty][tx].y = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_ONE * 2 < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH_ONE].x = __float2half_ru(temp_x3);
    } else {
      shmem_X[ty][tx + TILE_WIDTH_ONE].x = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_ONE * 3 < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH_ONE].y = __float2half_ru(temp_x4);
    } else {
      shmem_X[ty][tx + TILE_WIDTH_ONE].y = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_ONE * 4 < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH_ONE * 2].x = __float2half_ru(temp_x5);
    } else {
      shmem_X[ty][tx + TILE_WIDTH_ONE * 2].x = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_ONE * 5 < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH_ONE * 2].y = __float2half_ru(temp_x6);
    } else {
      shmem_X[ty][tx + TILE_WIDTH_ONE * 2].y = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_ONE * 6 < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH_ONE * 3].x = __float2half_ru(temp_x7);
    } else {
      shmem_X[ty][tx + TILE_WIDTH_ONE * 3].x = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_ONE * 7 < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH_ONE * 3].y = __float2half_ru(temp_x8);
    } else {
      shmem_X[ty][tx + TILE_WIDTH_ONE * 3].y = __float2half_rd(0.f);
    }

    __syncthreads();

    temp_col += TILE_WIDTH_ONE;
    temp_row += TILE_WIDTH_ONE;

    X_c = getC(temp_row);
    X_p = getK1(temp_row);
    X_q = getK2(temp_row);
    temp_k  = __float2half_rn(k[row * numMatACol + temp_col]);
    temp_x1 = x4d(bz, X_c, X_h1 + X_p, X_w1 + X_q);
    temp_x2 = x4d(bz, X_c, X_h2 + X_p, X_w2 + X_q);
    temp_x3 = x4d(bz, X_c, X_h3 + X_p, X_w3 + X_q);
    temp_x4 = x4d(bz, X_c, X_h4 + X_p, X_w4 + X_q);
    temp_x5 = x4d(bz, X_c, X_h5 + X_p, X_w5 + X_q);
    temp_x6 = x4d(bz, X_c, X_h6 + X_p, X_w6 + X_q);
    temp_x7 = x4d(bz, X_c, X_h7 + X_p, X_w7 + X_q);
    temp_x8 = x4d(bz, X_c, X_h8 + X_p, X_w8 + X_q);

    #pragma unroll
    for (int q = 0; q < TILE_WIDTH_ONE; ++q) {
      acc1 = __hfma2(shmem_K[ty][q], shmem_X[q][tx], acc1);
      acc2 = __hfma2(shmem_K[ty][q], shmem_X[q][tx + TILE_WIDTH_ONE], acc2);
      acc3 = __hfma2(shmem_K[ty][q], shmem_X[q][tx + TILE_WIDTH_ONE*2], acc3);
      acc4 = __hfma2(shmem_K[ty][q], shmem_X[q][tx + TILE_WIDTH_ONE*3], acc4);
    }

    __syncthreads();

    if (row < M && col < W_out * H_out)                      y4d(bz, row, X_h1, X_w1) = __low2float(acc1);
    if (row < M && col + TILE_WIDTH_ONE * 1 < W_out * H_out) y4d(bz, row, X_h2, X_w2) = __high2float(acc1);
    if (row < M && col + TILE_WIDTH_ONE * 2 < W_out * H_out) y4d(bz, row, X_h3, X_w3) = __low2float(acc2);
    if (row < M && col + TILE_WIDTH_ONE * 3 < W_out * H_out) y4d(bz, row, X_h4, X_w4) = __high2float(acc2);
    if (row < M && col + TILE_WIDTH_ONE * 4 < W_out * H_out) y4d(bz, row, X_h5, X_w5) = __low2float(acc3);
    if (row < M && col + TILE_WIDTH_ONE * 5 < W_out * H_out) y4d(bz, row, X_h6, X_w6) = __high2float(acc3);
    if (row < M && col + TILE_WIDTH_ONE * 6 < W_out * H_out) y4d(bz, row, X_h7, X_w7) = __low2float(acc4);
    if (row < M && col + TILE_WIDTH_ONE * 7 < W_out * H_out) y4d(bz, row, X_h8, X_w8) = __high2float(acc4);
  }
}


__global__ void 
__launch_bounds__(TILE_WIDTH_TWO * TILE_WIDTH_TWO, 2)
forward_shared_unroll_two(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out) {
  __shared__ half2 shmem_X[TILE_WIDTH_TWO][TILE_WIDTH_TWO * GRANULARITY / 2];
  __shared__ half2 shmem_K[TILE_WIDTH_TWO][TILE_WIDTH_TWO];

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x * GRANULARITY + tx;
  int numMatACol = C * K * K;

  half2 acc1 = __float2half2_rn(0.f);
  half2 acc2 = __float2half2_rn(0.f);
  half2 acc3 = __float2half2_rn(0.f);
  half2 acc4 = __float2half2_rn(0.f);

  int temp_col = tx;
  int temp_row = ty;

  int X_h1 = col / W_out;
  int X_w1 = col % W_out;

  int X_h2 = (col + TILE_WIDTH_TWO) / W_out;
  int X_w2 = (col + TILE_WIDTH_TWO) % W_out;

  int X_h3 = (col + TILE_WIDTH_TWO*2) / W_out;
  int X_w3 = (col + TILE_WIDTH_TWO*2) % W_out;

  int X_h4 = (col + TILE_WIDTH_TWO*3) / W_out;
  int X_w4 = (col + TILE_WIDTH_TWO*3) % W_out;

  int X_h5 = (col + TILE_WIDTH_TWO*4) / W_out;
  int X_w5 = (col + TILE_WIDTH_TWO*4) % W_out;

  int X_h6 = (col + TILE_WIDTH_TWO*5) / W_out;
  int X_w6 = (col + TILE_WIDTH_TWO*5) % W_out;

  int X_h7 = (col + TILE_WIDTH_TWO*6) / W_out;
  int X_w7 = (col + TILE_WIDTH_TWO*6) % W_out;

  int X_h8 = (col + TILE_WIDTH_TWO*7) / W_out;
  int X_w8 = (col + TILE_WIDTH_TWO*7) % W_out;

  int X_c  = getC(temp_row);
  int X_p  = getK1(temp_row);
  int X_q  = getK2(temp_row);

  half temp_k  = __float2half_ru(k[row * numMatACol + temp_col]);
  float temp_x1 = x4d(bz, X_c, X_h1 + X_p, X_w1 + X_q);
  float temp_x2 = x4d(bz, X_c, X_h2 + X_p, X_w2 + X_q);
  float temp_x3 = x4d(bz, X_c, X_h3 + X_p, X_w3 + X_q);
  float temp_x4 = x4d(bz, X_c, X_h4 + X_p, X_w4 + X_q);
  float temp_x5 = x4d(bz, X_c, X_h5 + X_p, X_w5 + X_q);
  float temp_x6 = x4d(bz, X_c, X_h6 + X_p, X_w6 + X_q);
  float temp_x7 = x4d(bz, X_c, X_h7 + X_p, X_w7 + X_q);
  float temp_x8 = x4d(bz, X_c, X_h8 + X_p, X_w8 + X_q);

  #pragma unroll
  for (int i = 0; i < (numMatACol + TILE_WIDTH_TWO - 1) / (TILE_WIDTH_TWO); ++i) {
    if (temp_col < numMatACol && row < M) {
      shmem_K[ty][tx] = __half2half2(temp_k);
    } else {
      shmem_K[ty][tx] = __float2half2_rn(0.f);
    }

    if (temp_row < numMatACol && col < H_out * W_out) {
      shmem_X[ty][tx].x = __float2half_rd(temp_x1);
    } else {
      shmem_X[ty][tx].x = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_TWO < H_out * W_out) {
      shmem_X[ty][tx].y = __float2half_rd(temp_x2);
    } else {
      shmem_X[ty][tx].y = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_TWO * 2 < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH_TWO].x = __float2half_rd(temp_x3);
    } else {
      shmem_X[ty][tx + TILE_WIDTH_TWO].x = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_TWO * 3 < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH_TWO].y = __float2half_rd(temp_x4);
    } else {
      shmem_X[ty][tx + TILE_WIDTH_TWO].y = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_TWO * 4 < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH_TWO * 2].x = __float2half_rd(temp_x5);
    } else {
      shmem_X[ty][tx + TILE_WIDTH_TWO * 2].x = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_TWO * 5 < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH_TWO * 2].y = __float2half_rd(temp_x6);
    } else {
      shmem_X[ty][tx + TILE_WIDTH_TWO * 2].y = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_TWO * 6 < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH_TWO * 3].x = __float2half_rd(temp_x7);
    } else {
      shmem_X[ty][tx + TILE_WIDTH_TWO * 3].x = __float2half_rd(0.f);
    }

    if (temp_row < numMatACol && col + TILE_WIDTH_TWO * 7 < H_out * W_out) {
      shmem_X[ty][tx + TILE_WIDTH_TWO * 3].y = __float2half_rd(temp_x8);
    } else {
      shmem_X[ty][tx + TILE_WIDTH_TWO * 3].y = __float2half_rd(0.f);
    }

    __syncthreads();

    temp_col += TILE_WIDTH_TWO;
    temp_row += TILE_WIDTH_TWO;
    X_c = getC(temp_row);
    X_p = getK1(temp_row);
    X_q = getK2(temp_row);
    temp_k  = __float2half_ru(k[row * numMatACol + temp_col]);
    temp_x1 = x4d(bz, X_c, X_h1 + X_p, X_w1 + X_q);
    temp_x2 = x4d(bz, X_c, X_h2 + X_p, X_w2 + X_q);
    temp_x3 = x4d(bz, X_c, X_h3 + X_p, X_w3 + X_q);
    temp_x4 = x4d(bz, X_c, X_h4 + X_p, X_w4 + X_q);
    temp_x5 = x4d(bz, X_c, X_h5 + X_p, X_w5 + X_q);
    temp_x6 = x4d(bz, X_c, X_h6 + X_p, X_w6 + X_q);
    temp_x7 = x4d(bz, X_c, X_h7 + X_p, X_w7 + X_q);
    temp_x8 = x4d(bz, X_c, X_h8 + X_p, X_w8 + X_q);

    // #pragma unroll
    for (int q = 0; q < TILE_WIDTH_TWO; ++q) {
      acc1 = __hfma2(shmem_K[ty][q], shmem_X[q][tx], acc1);
      acc2 = __hfma2(shmem_K[ty][q], shmem_X[q][tx + TILE_WIDTH_TWO], acc2);
      acc3 = __hfma2(shmem_K[ty][q], shmem_X[q][tx + TILE_WIDTH_TWO*2], acc3);
      acc4 = __hfma2(shmem_K[ty][q], shmem_X[q][tx + TILE_WIDTH_TWO*3], acc4);
    }

    __syncthreads();

    if (row < M && col < W_out * H_out) {
      y4d(bz, row, X_h1,  X_w1)  = __low2float(acc1);
    }

    if (row < M && col + TILE_WIDTH_TWO < W_out * H_out) {
      y4d(bz, row, X_h2, X_w2) = __high2float(acc1);
    }

    if (row < M && col + TILE_WIDTH_TWO * 2 < W_out * H_out) {
      y4d(bz, row, X_h3, X_w3) = __low2float(acc2);
    }

    if (row < M && col + TILE_WIDTH_TWO * 3 < W_out * H_out) {
      y4d(bz, row, X_h4, X_w4) = __high2float(acc2);
    }

    if (row < M && col + TILE_WIDTH_TWO * 4 < W_out * H_out) {
      y4d(bz, row, X_h5, X_w5) = __low2float(acc3);
    }

    if (row < M && col + TILE_WIDTH_TWO * 5 < W_out * H_out) {
      y4d(bz, row, X_h6, X_w6) = __high2float(acc3);
    }

    if (row < M && col + TILE_WIDTH_TWO * 6 < W_out * H_out) {
      y4d(bz, row, X_h7, X_w7) = __low2float(acc4);
    }

    if (row < M && col + TILE_WIDTH_TWO * 7 < W_out * H_out) {
      y4d(bz, row, X_h8, X_w8) = __high2float(acc4);
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
  if (M == 24) {
    dim3 gridDim(((H_out*W_out-1)/(TILE_WIDTH_TWO*GRANULARITY)+1), ((M-1)/TILE_WIDTH_TWO+1), B);
    dim3 blockDim(TILE_WIDTH_TWO, TILE_WIDTH_TWO, 1);
    forward_shared_unroll_two<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K, H_out, W_out);
  } else {
    dim3 gridDim(((H_out*W_out-1)/(TILE_WIDTH_ONE*GRANULARITY)+1), ((M-1)/TILE_WIDTH_ONE+1), B);
    dim3 blockDim(TILE_WIDTH_ONE, TILE_WIDTH_ONE, 1);
    forward_shared_unroll_one<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K, H_out, W_out);
  }
  // dim3 gridDim(ceil(1.0*H_out*W_out/TILE_WIDTH/GRANULARITY), ceil(1.0*M/TILE_WIDTH), B);
  // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  // forward_shared_unroll<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K, H_out, W_out);
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