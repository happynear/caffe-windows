#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/label_specific_add_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void LabelSpecificAddForward(const int n, const int dim, const Dtype* label,
                                                 Dtype* top_data, Dtype bias) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      if (top_data[index * dim + gt] > -bias) top_data[index * dim + gt] += bias;
    }
  }

  template <typename Dtype>
  void LabelSpecificAddLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);
    if (!transform_test_ && this->phase_ == TEST) return;

    if (anneal_bias_) {
      bias_ = bias_base_ + pow(((Dtype)1. + bias_gamma_ * iteration_), bias_power_) - (Dtype)1.;
      bias_ = std::max(bias_, bias_min_);
      bias_ = std::min(bias_, bias_max_);
      iteration_++;
    }
    if (top.size() == 2) {
      top[1]->mutable_cpu_data()[0] = bias_;
    }

    // NOLINT_NEXT_LINE(whitespace/operators)
    LabelSpecificAddForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, label_data, top_data, bias_);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  void LabelSpecificAddLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down,
                                                     const vector<Blob<Dtype>*>& bottom) {
    if (top[0] != bottom[0] && propagate_down[0]) {
      int count = bottom[0]->count();
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      caffe_copy(count, top_diff, bottom_diff);
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(LabelSpecificAddLayer);
}  // namespace caffe
