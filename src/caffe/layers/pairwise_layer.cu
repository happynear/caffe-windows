#include <cfloat>
#include <vector>

#include "caffe/layers/pairwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void ProdForward(const int M_, const int N_, const int K_,
                               const Dtype* bottom_data_a, const Dtype* bottom_data_b,
                               Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, M_ * N_ * K_) {
      int m = index / N_ / K_;
      int nk = index % (N_* K_);
      int n = nk / K_;
      int k = nk % K_;
      top_data[index] = bottom_data_a[m * K_ + k] * bottom_data_b[n * K_ + k];
    }
  }

  template <typename Dtype>
  __global__ void SumForward(const int M_, const int N_, const int K_,
                              const Dtype* bottom_data_a, const Dtype* bottom_data_b,
                              Dtype coeff0, Dtype coeff1,
                              Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, M_ * N_ * K_) {
      int m = index / N_ / K_;
      int nk = index % (N_* K_);
      int n = nk / K_;
      int k = nk % K_;
      top_data[index] = bottom_data_a[m * K_ + k] *coeff0 + bottom_data_b[n * K_ + k] * coeff1;
    }
  }

  template <typename Dtype>
  __global__ void MaxForward(const int M_, const int N_, const int K_,
                             const Dtype* bottom_data_a, const Dtype* bottom_data_b,
                             Dtype* top_data, int* mask) {
    CUDA_KERNEL_LOOP(index, M_ * N_ * K_) {
      int m = index / N_ / K_;
      int nk = index % (N_* K_);
      int n = nk / K_;
      int k = nk % K_;
      if (bottom_data_a[m * K_ + k] > bottom_data_b[n * K_ + k]) {
        top_data[index] = bottom_data_a[m * K_ + k];
        mask[index] = 0;
      }
      else {
        top_data[index] = bottom_data_b[n * K_ + k];
        mask[index] = 1;
      }
    }
  }

template <typename Dtype>
void PairwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  switch (op_) {
  case PairwiseParameter_PairwiseOp_PROD:
    // NOLINT_NEXT_LINE(whitespace/operators)
    ProdForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      M_, N_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_data);
    break;
  case PairwiseParameter_PairwiseOp_SUM:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SumForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      M_, N_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), coeffs_[0], coeffs_[1], top_data);
    break;
  case PairwiseParameter_PairwiseOp_MAX:
    mask = max_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      M_, N_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_data, mask);
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
__global__ void ProdBackward_a(const int M_, const int N_, const int K_,
                               const Dtype* bottom_data_a, const Dtype* bottom_data_b,
                               const Dtype* top_diff, Dtype* bottom_diff_a) {
  CUDA_KERNEL_LOOP(index, M_ * K_) {
    int m = index / K_;
    int k = index % K_;
    for (int n = 0; n < N_; n++) {
      bottom_diff_a[index] += top_diff[(m*N_ + n)*K_ + k] * bottom_data_b[n*K_ + k];
    }
  }
}

template <typename Dtype>
__global__ void ProdBackward_b(const int M_, const int N_, const int K_,
                               const Dtype* bottom_data_a, const Dtype* bottom_data_b,
                               const Dtype* top_diff, Dtype* bottom_diff_b) {
  CUDA_KERNEL_LOOP(index, N_ * K_) {
    int n = index / K_;
    int k = index % K_;
    for (int m = 0; m < M_; m++) {
      bottom_diff_b[index] += top_diff[(m*N_ + n)*K_ + k] * bottom_data_a[m*K_ + k];
    }
  }
}

template <typename Dtype>
__global__ void SumBackward_a(const int M_, const int N_, const int K_,
                              const Dtype* bottom_data_a, const Dtype* bottom_data_b,
                              Dtype coeff0, Dtype coeff1,
                              const Dtype* top_diff, Dtype* bottom_diff_a) {
  CUDA_KERNEL_LOOP(index, M_ * K_) {
    int m = index / K_;
    int k = index % K_;
    for (int n = 0; n < N_; n++) {
      bottom_diff_a[index] += top_diff[(m*N_ + n)*K_ + k] * coeff0;
    }
  }
}

template <typename Dtype>
__global__ void SumBackward_b(const int M_, const int N_, const int K_,
                               const Dtype* bottom_data_a, const Dtype* bottom_data_b,
                              Dtype coeff0, Dtype coeff1,
                               const Dtype* top_diff, Dtype* bottom_diff_b) {
  CUDA_KERNEL_LOOP(index, N_ * K_) {
    int n = index / K_;
    int k = index % K_;
    for (int m = 0; m < M_; m++) {
      bottom_diff_b[index] += top_diff[(m*N_ + n)*K_ + k] * coeff1;
    }
  }
}

template <typename Dtype>
__global__ void MaxBackward_a(const int M_, const int N_, const int K_,
                              const Dtype* bottom_data_a, const Dtype* bottom_data_b, const int* mask,
                              const Dtype* top_diff, Dtype* bottom_diff_a) {
  CUDA_KERNEL_LOOP(index, M_ * K_) {
    int m = index / K_;
    int k = index % K_;
    for (int n = 0; n < N_; n++) {
      if(mask[(m*N_ + n)*K_ + k] == 0) bottom_diff_a[index] += top_diff[(m*N_ + n)*K_ + k];
    }
  }
}

template <typename Dtype>
__global__ void MaxBackward_b(const int M_, const int N_, const int K_,
                              const Dtype* bottom_data_a, const Dtype* bottom_data_b, const int* mask,
                              const Dtype* top_diff, Dtype* bottom_diff_b) {
  CUDA_KERNEL_LOOP(index, N_ * K_) {
    int n = index / K_;
    int k = index % K_;
    for (int m = 0; m < M_; m++) {
      if (mask[(m*N_ + n)*K_ + k] == 1) bottom_diff_b[index] += top_diff[(m*N_ + n)*K_ + k];
    }
  }
}

template <typename Dtype>
void PairwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data_a = bottom[0]->gpu_data();
  const Dtype* bottom_data_b = bottom[1]->gpu_data();
  Dtype* bottom_diff_a = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff_b = bottom[1]->mutable_gpu_diff();
  
  if (propagate_down[0]) {
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff_a);
    switch (op_) {
    case PairwiseParameter_PairwiseOp_PROD:
      // NOLINT_NEXT_LINE(whitespace/operators)
      ProdBackward_a<Dtype> << <CAFFE_GET_BLOCKS(M_*K_), CAFFE_CUDA_NUM_THREADS >> >(
        M_, N_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_diff, bottom_diff_a);
      break;
    case PairwiseParameter_PairwiseOp_SUM:
      // NOLINT_NEXT_LINE(whitespace/operators)
      SumBackward_a<Dtype> << <CAFFE_GET_BLOCKS(M_*K_), CAFFE_CUDA_NUM_THREADS >> >(
        M_, N_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), coeffs_[0], coeffs_[1], top_diff, bottom_diff_a);
      break;
    case PairwiseParameter_PairwiseOp_MAX:
      mask = max_idx_.gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxBackward_a<Dtype> << <CAFFE_GET_BLOCKS(M_*K_), CAFFE_CUDA_NUM_THREADS >> >(
        M_, N_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), mask, top_diff, bottom_diff_a);
      break;
    default:
      LOG(FATAL) << "Unknown pairwise operation.";
    }
  }

  if (propagate_down[1]) {
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom_diff_b);
    switch (op_) {
    case PairwiseParameter_PairwiseOp_PROD:
      // NOLINT_NEXT_LINE(whitespace/operators)
      ProdBackward_b<Dtype> << <CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS >> >(
        M_, N_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_diff, bottom_diff_b);
      break;
    case PairwiseParameter_PairwiseOp_SUM:
      // NOLINT_NEXT_LINE(whitespace/operators)
      SumBackward_b<Dtype> << <CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS >> >(
        M_, N_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), coeffs_[0], coeffs_[1], top_diff, bottom_diff_b);
      break;
    case PairwiseParameter_PairwiseOp_MAX:
      mask = max_idx_.gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxBackward_b<Dtype> << <CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS >> >(
        M_, N_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), mask, top_diff, bottom_diff_b);
      break;
    default:
      LOG(FATAL) << "Unknown pairwise operation.";
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PairwiseLayer);

}  // namespace caffe
