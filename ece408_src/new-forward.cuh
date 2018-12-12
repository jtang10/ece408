
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define MAX_NUM_THREADS 1024
#define TILE_WIDTH_ONE 12
#define TILE_WIDTH_TWO 24
#define size_t unsigned int

namespace mxnet
{
namespace op
{
__global__ void forward_shared_unroll_one(float *y, const float *x, const float *k,
                               				const int B, const int M, const int C,
                               				const int H, const int W, const int K, const int H_out, const int W_out) {
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	__shared__ float shmem_X[TILE_WIDTH_ONE * TILE_WIDTH_ONE];
	__shared__ float shmem_K[TILE_WIDTH_ONE * TILE_WIDTH_ONE];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int numMatACol = C * K * K;

  float acc = 0;

  // #pragma unroll
  for (int i = 0; i < (numMatACol - 1) / TILE_WIDTH_ONE + 1; ++i) { // ceil
  	int temp_col = i * TILE_WIDTH_ONE + threadIdx.x;
  	int temp_row = i * TILE_WIDTH_ONE + threadIdx.y;

  	if (temp_col < numMatACol && row < M) { //  && row < M
      shmem_K[threadIdx.y * TILE_WIDTH_ONE + threadIdx.x] = k[row * numMatACol + temp_col];
  	} else {
  		shmem_K[threadIdx.y * TILE_WIDTH_ONE + threadIdx.x] = 0;
  	}

  	// int X_b = b;
  	int X_c = temp_row / (K * K);
  	int X_h = col / W_out;
  	int X_w = col % W_out;
  	int X_p = (temp_row % (K * K)) / K;
    int X_q = (temp_row % (K * K)) % K;

  	if (temp_row < numMatACol && col < H_out * W_out) {
  		shmem_X[threadIdx.y * TILE_WIDTH_ONE + threadIdx.x] = x4d(blockIdx.z, X_c, X_h + X_p, X_w + X_q);
  	} else {
  		shmem_X[threadIdx.y * TILE_WIDTH_ONE + threadIdx.x] = 0;
  	}

  	__syncthreads();

    #pragma unroll
  	for (int q = 0; q < TILE_WIDTH_ONE; ++q) {
  		acc += shmem_K[threadIdx.y * TILE_WIDTH_ONE + q] * shmem_X[q * TILE_WIDTH_ONE + threadIdx.x];
  	}

  	__syncthreads();

  	if (row < M && col < W_out * H_out) {
      y[blockIdx.z * (M * H_out * W_out) + row * (H_out * W_out) + col] = acc;
  	}
  }

  #undef y4d
  #undef x4d
  #undef k4d
}



__global__ void forward_shared_unroll_two(float *y, const float *x, const float *k,
                               				const int B, const int M, const int C,
                               				const int H, const int W, const int K, const int H_out, const int W_out) {
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	__shared__ float shmem_X[TILE_WIDTH_TWO * TILE_WIDTH_TWO];
	__shared__ float shmem_K[TILE_WIDTH_TWO * TILE_WIDTH_TWO];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int numMatACol = C * K * K;

  float acc = 0;

  // #pragma unroll
  for (int i = 0; i < (numMatACol - 1) / TILE_WIDTH_TWO + 1; ++i) { // ceil
  	int temp_col = i * TILE_WIDTH_TWO + threadIdx.x;
  	int temp_row = i * TILE_WIDTH_TWO + threadIdx.y;

  	if (temp_col < numMatACol && row < M) { //  && row < M
      shmem_K[threadIdx.y * TILE_WIDTH_TWO + threadIdx.x] = k[row * numMatACol + temp_col];
  	} else {
  		shmem_K[threadIdx.y * TILE_WIDTH_TWO + threadIdx.x] = 0;
  	}

  	// int X_b = b;
  	int X_c = temp_row / (K * K);
  	int X_h = col / W_out;
  	int X_w = col % W_out;
  	int X_p = (temp_row % (K * K)) / K;
    int X_q = (temp_row % (K * K)) % K;

  	if (temp_row < numMatACol && col < H_out * W_out) {
  		shmem_X[threadIdx.y * TILE_WIDTH_TWO + threadIdx.x] = x4d(blockIdx.z, X_c, X_h + X_p, X_w + X_q);
  	} else {
  		shmem_X[threadIdx.y * TILE_WIDTH_TWO + threadIdx.x] = 0;
  	}

  	__syncthreads();

    #pragma unroll
  	for (int q = 0; q < TILE_WIDTH_TWO; ++q) {
  		acc += shmem_K[threadIdx.y * TILE_WIDTH_TWO + q] * shmem_X[q * TILE_WIDTH_TWO + threadIdx.x];
  	}

  	__syncthreads();

  	if (row < M && col < W_out * H_out) {
      y[blockIdx.z * (M * H_out * W_out) + row * (H_out * W_out) + col] = acc;
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

  // printf("ceil(1.0*M/TILE_WIDTH): %f\n", ceil(1.0*M/TILE_WIDTH));
  // printf("TILE_WIDTH: %d\n", TILE_WIDTH);
  // printf("M: %d\n", M);
  // printf("B: %d\n", B);
  // printf("C: %d\n", C);
  // printf("H: %d\n", H);
  // printf("W: %d\n", W);
  // printf("K: %d\n", K);
  // ceil(1.0*M/TILE_WIDTH): 1.000000
  // // TILE_WIDTH: 12
  // // M: 12
  // // B: 10000
  // // C: 1
  // // H: 72
  // // W: 72
  // // K: 7
  // // Op Time: 0.037503
  // // ceil(1.0*M/TILE_WIDTH): 2.000000
  // // TILE_WIDTH: 12
  // // M: 24
  // // B: 10000
  // // C: 12
  // // H: 33
  // // W: 33
  // // K: 7

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

  if (M == 24) {
    dim3 gridDim(((H_out*W_out-1)/TILE_WIDTH_TWO+1), ((M-1)/TILE_WIDTH_TWO+1), B);
    dim3 blockDim(TILE_WIDTH_TWO, TILE_WIDTH_TWO, 1);
    forward_shared_unroll_two<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K, H_out, W_out);
  } else {
    dim3 gridDim(((H_out*W_out-1)/TILE_WIDTH_ONE+1), ((M-1)/TILE_WIDTH_ONE+1), B);
    dim3 blockDim(TILE_WIDTH_ONE, TILE_WIDTH_ONE, 1);
    forward_shared_unroll_one<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K, H_out, W_out);
  }
  // dim3 gridDim(ceil(1.0*H_out*W_out/TILE_WIDTH), 1, B); // M == TILE_WIDTH == 12
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
