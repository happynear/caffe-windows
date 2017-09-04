#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/label_specific_margin_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void ArccosForward(const int n, const Dtype* in, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      out[index] = Dtype(acos(in[index]) / M_PI * 180.0);
    }
  }

  template <typename Dtype>
  __global__ void CreateMask(const int num, const int dim, const Dtype* label, Dtype* positive_mask, Dtype* negative_mask) {
    CUDA_KERNEL_LOOP(index, num) {
      int gt = static_cast<int>(label[index]);
      positive_mask[index*dim + gt] = Dtype(1);
      negative_mask[index*dim + gt] = Dtype(0);
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificMarginForward(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                             Dtype* top_data, Dtype margin) {
    CUDA_KERNEL_LOOP(index, n) {
      int l = static_cast<int>(label[index]);
      top_data[index * dim + l] = bottom_data[index * dim + l] * cosf(margin / 180 * M_PI) -
        sqrt(1 - bottom_data[index * dim + l] * bottom_data[index * dim + l] + 1e-12) * sinf(margin / 180 * M_PI);
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificMarginBackward(const int n, const int dim, const Dtype* top_diff, const Dtype* label,
                                                     Dtype* bottom_diff, const Dtype* bottom_data, Dtype margin) {
    CUDA_KERNEL_LOOP(index, n) {
      int l = static_cast<int>(label[index]);
      bottom_diff[index * dim + l] = top_diff[index * dim + l] * (cosf(margin / 180 * M_PI) -
                                                    bottom_data[index * dim + l] / sqrt(1 - bottom_data[index * dim + l] * bottom_data[index * dim + l] + 1e-12) * sinf(margin / 180 * M_PI));
    }
  }

template <typename Dtype>
void LabelSpecificMarginLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* margin = this->blobs_[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  if (has_margin_base_) {
    margin[0] = margin_base_ + pow(((Dtype)1. + gamma_ * iter_), power_) - 1;
    iter_++;
  }
  if (has_margin_max_) {
    margin[0] = std::min(margin[0], margin_max_);
  }

  if (top.size() >= 2 && auto_tune_) {
    Dtype *positive_mask_data = positive_mask.mutable_gpu_data();
    Dtype *negative_mask_data = negative_mask.mutable_gpu_data();
    caffe_gpu_set(count, Dtype(0), positive_mask_data);
    caffe_gpu_set(count, Dtype(1), negative_mask_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    CreateMask<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, label_data, positive_mask.mutable_gpu_data(), negative_mask.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

    Dtype positive_mean;
    Dtype positive_std;
    Dtype negative_mean;
    Dtype negative_std;
    
    // NOLINT_NEXT_LINE(whitespace/operators)
    ArccosForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
      count, bottom_data, bottom_angle.mutable_gpu_data());
    caffe_gpu_powx(count, bottom_angle.gpu_data(), Dtype(2), bottom_square.mutable_gpu_data());
    caffe_gpu_dot(count, bottom_angle.gpu_data(), positive_mask.gpu_data(), &positive_mean);
    caffe_gpu_dot(count, bottom_square.gpu_data(), positive_mask.gpu_data(), &positive_std);
    caffe_gpu_dot(count, bottom_angle.gpu_data(), negative_mask.gpu_data(), &negative_mean);
    caffe_gpu_dot(count, bottom_square.gpu_data(), negative_mask.gpu_data(), &negative_std);

    positive_mean /= num;
    positive_std = sqrt(positive_std / num - positive_mean * positive_mean);
    negative_mean /= num * (dim - 1);
    negative_std = sqrt(negative_std / num / (dim - 1) - negative_mean * negative_mean);
    
    if (this->phase_ == TEST) {
      top[1]->mutable_cpu_data()[0] = margin[0];
      top[1]->mutable_cpu_data()[1] = positive_mean;
      top[1]->mutable_cpu_data()[2] = positive_std;
      top[1]->mutable_cpu_data()[3] = negative_mean;
      top[1]->mutable_cpu_data()[4] = negative_std;
    }
    else {
      if (iter_ == 1) {
        margin[1] = positive_mean;
        margin[2] = positive_std;
        margin[3] = negative_mean;
        margin[4] = negative_std;
      }
      else {
        margin[1] = 0.99 * margin[1] + 0.01 * positive_mean;
        margin[2] = 0.99 * margin[2] + 0.01 * positive_std;
        margin[3] = 0.99 * margin[3] + 0.01 * negative_mean;
        margin[4] = 0.99 * margin[4] + 0.01 * negative_std;
      }
     
      margin[0] = (margin[3] - margin[1]) / (margin[2] + margin[3]) * margin[2];
      caffe_copy(5, this->blobs_[0]->cpu_data(), top[1]->mutable_cpu_data());
    }
  }
  if (top.size() >= 2 && !auto_tune_) {
    top[1]->mutable_cpu_data()[0] = margin[0];
  }

  caffe_copy(count, bottom_data, top_data);
  if (!margin_on_test_ && this->phase_ == TEST) return;

  if (margin[0] != Dtype(0.0)) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    LabelSpecificMarginForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, bottom_data, label_data, top_data, margin[0]);
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
void LabelSpecificMarginLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* margin = this->blobs_[0]->mutable_cpu_data();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    caffe_copy(count, top_diff, bottom_diff);
    if (!margin_on_test_ && this->phase_ == TEST) return;
    
    if (margin[0] != Dtype(0.0)) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificMarginBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, top_diff, label_data, bottom_diff, bottom_data, margin[0]);
      CUDA_POST_KERNEL_CHECK;
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(LabelSpecificMarginLayer);


}  // namespace caffe
