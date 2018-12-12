
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <stdio.h>

#define MAX_NUM_THREADS 1024
#define TILE_WIDTH 16
#define TILE_HEIGHT 4
#define size_t unsigned int

// Maximum global memory size: 12618760192
// Maximum constant memory size: 65536
// Maximum shared memory size per block: 49152
// Maximum block dimensions: 1024 x 1024 x 64
// Maximum grid dimensions: 2147483647 x 65535 x 65535

// X: 100 X 1 X 72 X 72 matrix
// K: 12 X 1 X 7 X 7 matrix
// Y: 100 X 12 X 66 X 66 matrix

// X: 100 X 12 X 33 X 33 matrix
// K: 24 X 12 X 7 X 7 matrix
// Y: 100 X 24 X 27 X 27 matrix

namespace mxnet
{
namespace op
{
__global__ void forward_shared_unroll(float *y, const float *x, const float *k,
                               				const int B, const int M, const int C,
                               				const int H, const int W, const int K, const int H_out, const int W_out) {
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  __shared__ float shmem_X[TILE_HEIGHT][TILE_WIDTH];
  __shared__ float shmem_K[TILE_HEIGHT][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;
  int numMatACol = C * K * K;
  int b = blockIdx.z;

  float acc = 0;

  // #pragma unroll
  for (int i = 0; i < ceil(1.0 * numMatACol / TILE_WIDTH); ++i) {
  	int temp_col = i * TILE_WIDTH + tx;
  	int temp_row = i * TILE_HEIGHT + ty;

  	int K_m = row;
  	int K_c = temp_col / (K * K);
  	int K_k1 = (temp_col % (K * K)) / K;
  	int K_k2 = (temp_col % (K * K)) % K;

  	if (temp_col < numMatACol && row < M) {
  		shmem_K[ty][tx] = k4d(K_m, K_c, K_k1, K_k2);
  	} else {
  		shmem_K[ty][tx] = 0;
  	}

  	int X_b = b;
  	int X_c = temp_row / (K * K);
  	int X_h = col / W_out;
  	int X_w = col % W_out;
  	int X_p = (temp_row % (K * K)) / K;
    int X_q = (temp_row % (K * K)) % K;

  	if (temp_row < numMatACol && col < H_out * W_out) {
  		shmem_X[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
  	} else {
  		shmem_X[ty][tx] = 0;
  	}

  	__syncthreads();

    // #pragma unroll
  	for (int q = 0; q < TILE_WIDTH; ++q) {
  		acc += shmem_K[ty][q] * shmem_X[q][tx];
  	}

  	__syncthreads();

  	int Y_b = b;
  	int Y_m = row;
  	int Y_h = col / W_out;
  	int Y_w = col % W_out;

  	if (row < M && col < W_out * H_out) {
      	y4d(Y_b, Y_m, Y_h, Y_w) = acc;
  	}
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



// code to get device props:
// int device;
// cudaGetDevice(&device);
// cudaDeviceProp props;
// cudaGetDeviceProperties(&props, device);
//
// printf(" Maximum global memory size: %zu\n",
//       props.totalGlobalMem);
// printf(" Maximum constant memory size: %zu\n",
//       props.totalConstMem);
// printf(" Maximum shared memory size per block: %zu\n",
//       props.sharedMemPerBlock);
// printf(" Maximum block dimensions: %d x %d x %d\n",
//       props.maxThreadsDim[0],
//       props.maxThreadsDim[1],
//       props.maxThreadsDim[2]);
// printf(" Maximum grid dimensions: %d x %d x %d\n",
//       props.maxGridSize[0],
//       props.maxGridSize[1],
//       props.maxGridSize[2]);
// printf(" Warp size: %d\n", props.warpSize);
//

#endif
