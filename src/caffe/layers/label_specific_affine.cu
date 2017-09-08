#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/label_specific_affine_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void LabelSpecificAffineForward(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                                 Dtype* top_data, Dtype scale, Dtype bias) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      top_data[index * dim + gt] = bottom_data[index * dim + gt] * scale + bias;
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificAffineBackward(const int n, const int dim, const Dtype* top_diff, const Dtype* label,
                                              Dtype* bottom_diff, Dtype scale) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      bottom_diff[index * dim + gt] = top_diff[index * dim + gt] * scale;
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificAffineBackwardScale(const int n, const int dim, const Dtype* top_diff, const Dtype* label,
                                                   const Dtype* bottom_data, Dtype* selected_value) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      selected_value[index] = top_diff[index * dim + gt] * bottom_data[index * dim + gt];
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificAffineBackwardBias(const int n, const int dim, const Dtype* top_diff, const Dtype* label,
                                                  const Dtype* bottom_data, Dtype* selected_value) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      selected_value[index] = top_diff[index * dim + gt];
    }
  }

  template <typename Dtype>
  void LabelSpecificAffineLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* scale_bias = (bottom.size() == 3) ? bottom[2]->cpu_data() : this->blobs_[0]->cpu_data();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    if (this->phase_ == TEST) {
      scale = Dtype(1);
      bias = Dtype(0);
    }
    else {
      if (auto_tune_) {
        scale = scale_bias[0];
        bias = scale_bias[1];
      }
      else {
        scale = scale_base_ * pow(((Dtype)1. + scale_gamma_ * iteration_), scale_power_);
        bias = bias_base_ + pow(((Dtype)1. + bias_gamma_ * iteration_), bias_power_) - (Dtype)1.;
        scale = std::min(scale, scale_max_);
        bias = std::min(bias, bias_max_);
        iteration_++;
      }
    }

    caffe_copy(count, bottom_data, top_data);
    if (!transform_test_ && this->phase_ == TEST) return;

    // NOLINT_NEXT_LINE(whitespace/operators)
    LabelSpecificAffineForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, bottom_data, label_data, top_data, scale, bias);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  void LabelSpecificAffineLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down,
                                                     const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* scale_bias_diff = bottom.size() == 3 ? bottom[2]->mutable_cpu_diff() : this->blobs_[0]->mutable_cpu_diff();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    if (propagate_down[0]) {
      caffe_copy(count, top_diff, bottom_diff);
      if (!transform_test_ && this->phase_ == TEST) return;

      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificAffineBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, top_diff, label_data, bottom_diff, scale);
      CUDA_POST_KERNEL_CHECK;
    }

    if (auto_tune_ || (bottom.size() == 3 && propagate_down[2]) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificAffineBackwardScale<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, top_diff, label_data, bottom_data, selected_value_.mutable_gpu_data());
      caffe_gpu_dot(num, selected_value_.gpu_data(), sum_multiplier_.gpu_data(), scale_bias_diff);

      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificAffineBackwardBias<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, top_diff, label_data, bottom_data, selected_value_.mutable_gpu_data());
      caffe_gpu_dot(num, selected_value_.gpu_data(), sum_multiplier_.gpu_data(), scale_bias_diff+1);
    }
  }


  INSTANTIATE_LAYER_GPU_FUNCS(LabelSpecificAffineLayer);


}  // namespace caffe
