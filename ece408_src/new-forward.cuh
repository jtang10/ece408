
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define MAX_NUM_THREADS 1024
#define TILE_WIDTH 16

namespace mxnet 
{
namespace op 
{
// __global__ void forward_unroll(float *y, const float *x, const float *k, 
//                                const int B, const int M, const int C, 
//                                const int H, const int W, const int K) {
//   const int H_out = H - K + 1;
//   const int W_out = W - K + 1;

//   #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//   #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//   #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  
//   int bx = blockIdx.x;  int by = blockIdx.y;
//   int tx = threadIdx.x; int ty = threadIdx.y;
  
//   const int idx_x = blockDim.x * bx + tx;
//   const int idx_y = blockDim.y * by + ty;

//   // const int H_grid = ceil(B * H_out * 1.0 / TILE_WIDTH);
//   // const int W_grid = ceil(M * W_out * 1.0 / TILE_WIDTH);
//   // dim3 gridDim(W_grid, H_grid);
//   // dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
//   const int b = idx_y / H_out;
//   const int m = idx_x / W_out;
//   const int h = idx_y % H_out;
//   const int w = idx_x % W_out;

//   if (idx_x < W_out * M && idx_y < H_out * B) {
//     float acc = 0;
//     for(int c = 0; c < C; ++c) {
//       for(int p = 0; p < K; ++p) {
//         for(int q = 0; q < K; ++q) {
//           acc += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
//         }
//       }
//     }
//     for (int i = 0; i < C*K*K; ++i) {
//       int index_k_base = 
//       acc += 
//     }
//     y4d(b, m, h, w) = acc;
//   }

//   #undef y4d
//   #undef x4d
//   #undef k4d 
// }

__constant__ float Weights[14112];

__global__ void forward_m4(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_grid, const int W_grid) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]
    #define w4d(i3,i2,i1,i0) Weights[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    int X_tile_width = TILE_WIDTH + K - 1;
    extern __shared__ float shmem[];
    float *X_shared = &shmem[0];

    int n, m, h, w;
    int h0, w0, h_base, w_base;

    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.x;
    w0 = threadIdx.y;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h = h_base + h0;
    w = w_base + w0;

    float acc = 0;
    #pragma unroll
    for (int c = 0; c < C; c++) {
        #pragma unroll
        for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
            #pragma unroll
            for(int j = w; j < w_base + X_tile_width; j += TILE_WIDTH){
                if((i - h_base) < X_tile_width && (j-w_base) < X_tile_width){
                  X_shared[(i - h_base) * X_tile_width + j - w_base] = x4d(n,c,i,j);
                }
                else{
                  X_shared[(i - h_base) * X_tile_width + j - w_base] = 0;
                }
            }
         }

        __syncthreads();

        #pragma unroll
        for (int p = 0; p < K; p++) {
            #pragma unroll
            for(int q = 0; q < K; q++) {
                acc += X_shared[(h0+p) * X_tile_width + w0+q] * w4d(m, c, p, q);
            }
        }

        __syncthreads();
    }

    if(n < B && m < M && h < H_out && w < W_out) {
      y4d(n,m,h,w) = acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
}

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

    #pragma unroll
    for (p = 0; p < K; ++p) {
      #pragma unroll
      for (q = 0; q < K; ++q) {
        x_col[c*H_col*W_col + (p*K + q)*W_col + s] = x[c * H * W + (h_out + p) * W + (w_out + q)];
      }
    }
  }
}

__global__ void matrixMultiplykernel(float* k, float* x, float* y, int M, int H, int W){
  __shared__ float sharedW[TILE_WIDTH][TILE_WIDTH];
  __shared__ float sharedX[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;

  float Pvalue = 0;
  #pragma unroll
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

      #pragma unroll
      for(int i = 0; i < TILE_WIDTH; ++i) {
        Pvalue += sharedW[ty][i] * sharedX[i][tx];
      }

      __syncthreads();
  }

  if(Row < M && Col < W){
    y[Row * W + Col] = Pvalue;
  }
}

__global__ void forward_kernel1(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

  /*
  Modify this function to implement the forward pass described in Chapter 16.
  We have added an additional dimension to the tensors to support an entire mini-batch
  The goal here is to be correct AND fast.
  We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.

  x: B C H W
  y: B M H W
  k: M C K K

  X: 10000 X 1 X 72 X 72 matrix
  K: 12 X 1 X 7 X 7 matrix
  Y: 10000 X 12 X 66 X 66 matrix
  Op Time: 0.284250
  X: 10000 X 12 X 33 X 33 matrix
  K: 24 X 12 X 7 X 7 matrix
  Y: 10000 X 24 X 27 X 27 matrix
  Op Time: 0.466011

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
  // printf("X: %u X %u X %u X %u matrix\n", x.size(0), x.size(1), x.size(2), x.size(3));
  // printf("K: %u X %u X %u X %u matrix\n", w.size(0), w.size(1), w.size(2), w.size(3));
  // printf("Y: %u X %u X %u X %u matrix\n", y.size(0), y.size(1), y.size(2), y.size(3));

  const int B = x.shape_[0];
  const int M = y.shape_[1];
  const int C = x.shape_[1];
  const int H = x.shape_[2];
  const int W = x.shape_[3];
  const int K = w.shape_[3];

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  // Unrolling and matmul v1
  // float *x_unrolled;
  
  // const int H_unroll = C*K*K;
  // const int W_unroll = W_out * H_out;
  // cudaMalloc((void**) &x_unrolled, H_unroll * W_unroll * sizeof(float));

  // dim3 gridDim_unroll(ceil(1.0 * C * H_out * W_out / MAX_NUM_THREADS));
  // dim3 blockDim_unroll(MAX_NUM_THREADS);

  // dim3 grid_Multiple(ceil(H_out*W_out/(TILE_WIDTH*1.0)), (M - 1)/TILE_WIDTH + 1, 1);
  // dim3 block_Multiple(TILE_WIDTH, TILE_WIDTH, 1);

  // for (int i = 0; i < B; ++i) {
  //   unrollKernel<<<gridDim_unroll, blockDim_unroll>>>(x.dptr_ + C*H*W*i, x_unrolled, C, H, W, K);
  //   matrixMultiplykernel<<<grid_Multiple, block_Multiple>>>(w.dptr_, x_unrolled, y.dptr_ + M*H_out*W_out*i,  
  //                                                           M, H_unroll, W_unroll);
  // }

  // Kernel based on m3, for m4 only
  int W_grid = ceil(W_out/(float)TILE_WIDTH);
  int H_grid = ceil(H_out/(float)TILE_WIDTH);
  int Z = W_grid * H_grid;


  size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);
  int size = (M == 12 ? 588 : 14112);
  cudaMemcpyToSymbol(Weights, w.dptr_, size * sizeof(float), 0, cudaMemcpyHostToDevice);

  dim3 gridDim(B, M, Z);
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

  // Call the kernel
  forward_m4<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K, H_grid, W_grid);


  // Vanilla kernel
  // const int H_grid = ceil(B * H_out * 1.0 / TILE_WIDTH);
  // const int W_grid = ceil(M * W_out * 1.0 / TILE_WIDTH);
  // dim3 gridDim(W_grid, H_grid);
  // dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  // forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

  // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
  MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
  // cudaFree(x_unrolled);
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