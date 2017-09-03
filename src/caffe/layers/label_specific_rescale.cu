#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/label_specific_rescale_layer.hpp"

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif

namespace caffe {

  template <typename Dtype>
  __global__ void LabelSpecificPowerPositiveForward(const int n, const int dim, const Dtype* in, const Dtype* label,
                                               Dtype* out, Dtype positive_weight, bool bias_fix) {
    CUDA_KERNEL_LOOP(index, n) {
      int l = static_cast<int>(label[index]);
        /*out[index * dim + l] = (Dtype(0) < in[index * dim + l]) - (in[index * dim + l] < Dtype(0));
        out[index * dim + l] *= powf(abs(in[index * dim + l]), positive_weight);*/
      out[index * dim + l] = in[index * dim + l] * cosf(positive_weight / 180 * M_PI) - 
        sqrt(1 - in[index * dim + l] * in[index * dim + l] + 1e-12) * sinf(positive_weight / 180 * M_PI);
      if (bias_fix) {
        out[index * dim + l] *= positive_weight - 1;
        out[index * dim + l] -= positive_weight - 1;
        out[index * dim + l] += in[index * dim + l];
      }
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificPowerPositiveBackward(const int n, const int dim, const Dtype* in, const Dtype* label,
                                                    Dtype* out, const Dtype* bottom_data, Dtype positive_weight, bool bias_fix) {
    CUDA_KERNEL_LOOP(index, n) {
      int l = static_cast<int>(label[index]);
        /*out[index * dim + l] =
          in[index * dim + l] * positive_weight * powf(abs(bottom_data[index * dim + l]), positive_weight - 1);*/
      out[index * dim + l] = in[index * dim + l] * (cosf(positive_weight / 180 * M_PI) -
        bottom_data[index * dim + l] / sqrt(1 - bottom_data[index * dim + l] * bottom_data[index * dim + l] + 1e-12) * sinf(positive_weight / 180 * M_PI));
      if (bias_fix) {
        out[index * dim + l] *= positive_weight - 1;
        out[index * dim + l] += 1;
      }
    }
  }

template <typename Dtype>
__global__ void LabelSpecificRescalePositiveForward(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                            Dtype* top_data, Dtype positive_weight, bool for_ip, bool bias_fix) {
  CUDA_KERNEL_LOOP(index, n) {
    top_data[index * dim + static_cast<int>(label[index])] = bottom_data[index * dim + static_cast<int>(label[index])];
    if ((!for_ip) || bottom_data[index * dim + static_cast<int>(label[index])] > 0) {
      top_data[index * dim + static_cast<int>(label[index])] *= positive_weight;
      if (bias_fix) {
        top_data[index * dim + static_cast<int>(label[index])] += 1 - positive_weight;
      }
    }
  }
}

template <typename Dtype>
__global__ void LabelSpecificRescalePositiveBackward(const int n, const int dim, const Dtype* top_diff, const Dtype* label,
  Dtype* bottom_diff, const Dtype* bottom_data, Dtype positive_weight, bool for_ip) {
  CUDA_KERNEL_LOOP(index, n) {
    bottom_diff[index * dim + static_cast<int>(label[index])] = top_diff[index * dim + static_cast<int>(label[index])];
    if ((!for_ip) || bottom_data[index * dim + static_cast<int>(label[index])] > 0) {
      bottom_diff[index * dim + static_cast<int>(label[index])] *= positive_weight;
    }
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

  if (scale_positive_weight_) {
    iter_++;
    //lambda_ = base_ * pow(((Dtype)1. + gamma_ * iter_), -power_);
    positive_weight = positive_weight_base_ * pow(((Dtype)1. + gamma_ * iter_), -power_);
    if (has_positive_weight_min_) {
      positive_weight = std::max(positive_weight, positive_weight_min_);
    }
    if (has_positive_weight_max_) {
      positive_weight = std::min(positive_weight, positive_weight_max_);
    }
  }
  if (top.size() >= 2)top[1]->mutable_cpu_data()[0] = positive_weight;

  if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);
  if (!rescale_test && this->phase_ == TEST) return;

  if (positive_weight != Dtype(1.0)) {
    if (power_on_positive_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificPowerPositiveForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >(
        num, dim, bottom_data, label_data, top_data, positive_weight, bias_fix_);
      CUDA_POST_KERNEL_CHECK;
    }
    else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificRescalePositiveForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >(
        num, dim, bottom_data, label_data, top_data, positive_weight, for_ip, bias_fix_);
      CUDA_POST_KERNEL_CHECK;
    }
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
      if (power_on_positive_) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        LabelSpecificPowerPositiveBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >(
          num, dim, top_diff, label_data, bottom_diff, bottom_data, positive_weight, bias_fix_);
        CUDA_POST_KERNEL_CHECK;
      }
      else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        LabelSpecificRescalePositiveBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
          num, dim, top_diff, label_data, bottom_diff, bottom_data, positive_weight, for_ip);
        CUDA_POST_KERNEL_CHECK;
      }
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
