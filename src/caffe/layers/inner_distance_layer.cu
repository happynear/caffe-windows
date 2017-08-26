#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_distance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define sign(x) (Dtype(0) < (x)) - ((x) < Dtype(0))

  template <typename Dtype>
  __global__ void kernel_channel_dot(const int num, const int dim,
                                     const Dtype* data_1, const Dtype* data_2,
                                     Dtype* channel_dot) {
    CUDA_KERNEL_LOOP(index, num) {
      Dtype dot = 0;
      for (int d = 0; d < dim; ++d) {
        dot += data_1[index * dim + d] * data_2[index * dim + d];
      }
      channel_dot[index] = dot;
    }
  }

  template <typename Dtype>
  __global__ void kernel_channel_scal(const int num, const int dim,
                                      const Dtype* norm_data,
                                      Dtype* input_output_data) {
    CUDA_KERNEL_LOOP(index, num * dim) {
      int n = index / dim;
      input_output_data[index] *= norm_data[n];
    }
  }

  template <typename Dtype>
  __global__ void inner_distance_forward_L2(const int M_, const int N_, const int K_,
                                     const Dtype* bottom_data, const Dtype* weight, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, M_ * N_) {
      int m = index / N_;
      int n = index % N_;
      Dtype sum = Dtype(0);
      for (int k = 0; k < K_; ++k) {
        sum += (bottom_data[m * K_ + k] - weight[n * K_ + k]) * (bottom_data[m * K_ + k] - weight[n * K_ + k]);
      }
      top_data[index] = sum;
    }
  }

  template <typename Dtype>
  __global__ void inner_distance_forward_L1(const int M_, const int N_, const int K_,
                                            const Dtype* bottom_data, const Dtype* weight, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, M_ * N_) {
      int m = index / N_;
      int n = index % N_;
      Dtype sum = Dtype(0);
      for (int k = 0; k < K_; ++k) {
        sum += abs(bottom_data[m * K_ + k] - weight[n * K_ + k]);
      }
      top_data[index] = sum;
    }
  }

template <typename Dtype>
void InnerDistanceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = bottom.size() >= 2 ? bottom[1]->gpu_data() : this->blobs_[0]->gpu_data();

  if (normalize_ && bottom.size() == 1) {
    Dtype* mutable_weight = this->blobs_[0]->mutable_gpu_data();
    Dtype* weight_norm_data = weight_norm_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_dot<Dtype> << <CAFFE_GET_BLOCKS(N_),
      CAFFE_CUDA_NUM_THREADS >> > (N_, K_, weight, weight, weight_norm_data);
    caffe_gpu_powx(N_, weight_norm_data, Dtype(-0.5), weight_norm_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_scal<Dtype> << <CAFFE_GET_BLOCKS(N_ * K_),
      CAFFE_CUDA_NUM_THREADS >> > (N_, K_, weight_norm_data, mutable_weight);
  }

  if (distance_type_ == "L2") {
    // NOLINT_NEXT_LINE(whitespace/operators)
    inner_distance_forward_L2<Dtype> <<<CAFFE_GET_BLOCKS(M_ * N_),
      CAFFE_CUDA_NUM_THREADS >>> (M_, N_, K_,
                                  bottom_data, weight, top_data);
  }
  else if (distance_type_ == "L1") {
    // NOLINT_NEXT_LINE(whitespace/operators)
    inner_distance_forward_L1<Dtype> <<<CAFFE_GET_BLOCKS(M_ * N_),
      CAFFE_CUDA_NUM_THREADS >>> (M_, N_, K_,
                                  bottom_data, weight, top_data);
  }
  else {
    NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
__global__ void inner_distance_backward_L2(const int M_, const int N_, const int K_,
                                           const Dtype* bottom_data, const Dtype* weight, const Dtype* top_diff,
                                           Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, M_ * K_) {
    int m = index / K_;
    int k = index % K_;
    for (int n = 0; n < N_; ++n) {
      bottom_diff[index] += top_diff[m * N_ + n] * (bottom_data[m * K_ + k] - weight[n * K_ + k]) * Dtype(2);
    }
  }
}

template <typename Dtype>
__global__ void inner_distance_backward_L1(const int M_, const int N_, const int K_,
                                          const Dtype* bottom_data, const Dtype* weight, const Dtype* top_diff,
                                           Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, M_ * K_) {
    int m = index / K_;
    int k = index % K_;
    for (int n = 0; n < N_; ++n) {
      bottom_diff[index] += top_diff[m * N_ + n] * sign(bottom_data[m * K_ + k] - weight[n * K_ + k]);
    }
  }
}

template <typename Dtype>
__global__ void inner_distance_weight_backward_L2(const int M_, const int N_, const int K_,
                                           const Dtype* bottom_data, const Dtype* weight, const Dtype* top_diff,
                                           Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, N_ * K_) {
    int n = index / K_;
    int k = index % K_;
    for (int m = 0; m < M_; ++m) {
      weight_diff[index] += top_diff[m * N_ + n] * (weight[index] - bottom_data[m * K_ + k]) * Dtype(2);
    }
  }
}

template <typename Dtype>
__global__ void inner_distance_weight_backward_L2_center_only(const int M_, const int N_, const int K_,
                                                  const Dtype* bottom_data, const Dtype* weight,
                                                  const Dtype* label_data, const Dtype* top_diff,
                                                  Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, K_) {
    int k = index;
    for (int m = 0; m < M_; ++m) {
      int n = static_cast<int>(label_data[m]);
      weight_diff[n * K_ + k] += top_diff[m * N_ + n] * (weight[n * K_ + k] - bottom_data[m * K_ + k]) * Dtype(2);
    }
  }
}

template <typename Dtype>
__global__ void inner_distance_weight_backward_L1(const int M_, const int N_, const int K_,
                                                  const Dtype* bottom_data, const Dtype* weight, const Dtype* top_diff,
                                                  Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, N_ * K_) {
    int n = index / K_;
    int k = index % K_;
    for (int m = 0; m < M_; ++m) {
      weight_diff[index] += top_diff[m * N_ + n] * sign(weight[index] - bottom_data[m * K_ + k]);
    }
  }
}

template <typename Dtype>
__global__ void inner_distance_weight_backward_L1_center_only(const int M_, const int N_, const int K_,
                                                              const Dtype* bottom_data, const Dtype* weight,
                                                              const Dtype* label_data, const Dtype* top_diff,
                                                              Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, K_) {
    int k = index;
    for (int m = 0; m < M_; ++m) {
      int n = static_cast<int>(label_data[m]);
      weight_diff[n * K_ + k] += top_diff[m * N_ + n] * sign(weight[n * K_ + k] - bottom_data[m * K_ + k]);
    }
  }
}

template <typename Dtype>
void InnerDistanceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = bottom.size() >= 2 ? bottom[1]->gpu_data() : this->blobs_[0]->gpu_data();

  if ((bottom.size() == 1 && this->param_propagate_down_[0]) ||
    (bottom.size() >= 2 && propagate_down[1])) {
    Dtype* weight_diff = bottom.size() >= 2 ? bottom[1]->mutable_gpu_diff() : this->blobs_[0]->mutable_gpu_diff();
    if (bottom.size() >= 2) {
      caffe_gpu_set(bottom[1]->count(), Dtype(0), weight_diff);
    }
    const Dtype* label_data = NULL;
    if (update_center_only_) {
      label_data = bottom[bottom.size() - 1]->gpu_data();
    }
    // Gradient with respect to weight
    if (distance_type_ == "L2") {
      if (update_center_only_) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        inner_distance_weight_backward_L2_center_only<Dtype> << <CAFFE_GET_BLOCKS(K_),
          CAFFE_CUDA_NUM_THREADS >> > (M_, N_, K_,
                                       bottom_data, weight, label_data, top_diff, weight_diff);
      }
      else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        inner_distance_weight_backward_L2<Dtype> << <CAFFE_GET_BLOCKS(N_ * K_),
          CAFFE_CUDA_NUM_THREADS >> > (M_, N_, K_,
                                       bottom_data, weight, top_diff, weight_diff);
      }
    }
    else if (distance_type_ == "L1") {
      if (update_center_only_) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        inner_distance_weight_backward_L1_center_only<Dtype> << <CAFFE_GET_BLOCKS(K_),
          CAFFE_CUDA_NUM_THREADS >> > (M_, N_, K_,
                                       bottom_data, weight, label_data, top_diff, weight_diff);
      }
      else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        inner_distance_weight_backward_L1<Dtype> << <CAFFE_GET_BLOCKS(N_ * K_),
          CAFFE_CUDA_NUM_THREADS >> > (M_, N_, K_,
                                       bottom_data, weight, top_diff, weight_diff);
      }
    }
    else {
      NOT_IMPLEMENTED;
    }
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set<Dtype>(M_ * K_, 0, bottom_diff);
    if (distance_type_ == "L2") {
      // NOLINT_NEXT_LINE(whitespace/operators)
      inner_distance_backward_L2<Dtype> << <CAFFE_GET_BLOCKS(M_ * K_),
        CAFFE_CUDA_NUM_THREADS >> > (M_, N_, K_,
                                     bottom_data, weight, top_diff, bottom_diff);
    }
    else if (distance_type_ == "L1") {
      // NOLINT_NEXT_LINE(whitespace/operators)
      inner_distance_backward_L1<Dtype> << <CAFFE_GET_BLOCKS(M_ * K_),
        CAFFE_CUDA_NUM_THREADS >> > (M_, N_, K_,
                                     bottom_data, weight, top_diff, bottom_diff);
    }
    else {
      NOT_IMPLEMENTED;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerDistanceLayer);

}  // namespace caffe
