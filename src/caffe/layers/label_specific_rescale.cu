#include <algorithm>
#include <vector>

#include "caffe/layers/label_specific_rescale_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LabelSpecificRescalePositive(const int n, const int dim, const Dtype* in, const Dtype* label,
                                            Dtype* out, Dtype positive_weight, bool for_ip, float lambda) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index * dim + static_cast<int>(label[index])] = in[index * dim + static_cast<int>(label[index])] * (positive_weight + lambda);
    if (for_ip) out[index * dim + static_cast<int>(label[index])] += Dtype(1.0) - positive_weight;
    out[index * dim + static_cast<int>(label[index])] /= Dtype(1.0) + lambda;
  }
}

template <typename Dtype>
__global__ void LabelSpecificRescaleNegative(const int count, const int dim, const Dtype* in, const Dtype* label,
                                             Dtype* out, Dtype negative_weight) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / dim;
    int d = index % dim;
    if (d != static_cast<int>(label[n])) {
      out[n * dim + d] = in[n * dim + d] * negative_weight;
    }
  }
}

template <typename Dtype>
void LabelSpecificRescaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  iter_ += (Dtype)1.;
  lambda_ = base_ * pow(((Dtype)1. + gamma_ * iter_), -power_);
  lambda_ = std::max(lambda_, lambda_min_);
  if (top.size() >= 2)top[1]->mutable_gpu_data()[0] = lambda_;

  if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);
  if (!rescale_test && this->phase_ == TEST) return;

  if (positive_weight != Dtype(1.0)) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    LabelSpecificRescalePositive<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >(
      num, dim, bottom_data, label_data, top_data, positive_weight, for_ip, lambda_);
    CUDA_POST_KERNEL_CHECK;
  }
  if (negative_weight != Dtype(1.0)) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    LabelSpecificRescaleNegative<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      count, dim, bottom_data, label_data, top_data, negative_weight);
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
void LabelSpecificRescaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    if (top[0] != bottom[0]) caffe_copy(count, top_diff, bottom_diff);
    if (!rescale_test && this->phase_ == TEST) return;
    
    if (positive_weight != Dtype(1.0)) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificRescalePositive<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, top_diff, label_data, bottom_diff, positive_weight, false, lambda_);
      CUDA_POST_KERNEL_CHECK;
    }
    if (negative_weight != Dtype(1.0)) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificRescaleNegative<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
        count, dim, top_diff, label_data, bottom_diff, negative_weight);
      CUDA_POST_KERNEL_CHECK;
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(LabelSpecificRescaleLayer);


}  // namespace caffe
