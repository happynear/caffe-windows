#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/label_specific_margin_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void CreateMask(const int num, const int dim, const Dtype* bottom_data, const Dtype* label, Dtype* positive_mask, Dtype* negative_mask) {
    CUDA_KERNEL_LOOP(index, num * dim) {
      int n = index / dim;
      int d = index % dim;
      if (d == static_cast<int>(label[n])) {
        positive_mask[n*dim + d] = 1;
        negative_mask[n*dim + d] = 0;
      }
      else {
        positive_mask[n*dim + d] = 0;
        negative_mask[n*dim + d] = 1;
      }
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

  if (top.size() >= 2) {
    top[1]->mutable_cpu_data()[0] = margin[0];

    Blob<Dtype> positive_mask, negative_mask;
    positive_mask.ReshapeLike(*bottom[0]);
    negative_mask.ReshapeLike(*bottom[0]);
    // NOLINT_NEXT_LINE(whitespace/operators)
    CreateMask<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, bottom_data, label_data, positive_mask.mutable_gpu_data(), negative_mask.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

    Blob<Dtype> statistics;
    statistics.Reshape({ 4 });//positive_mean, positive_std, negative_mean, negative_std
    Blob<Dtype> bottom_square;
    bottom_square.ReshapeLike(*bottom[0]);
    caffe_gpu_powx(count, bottom_data, Dtype(2), bottom_square.mutable_gpu_data());
    caffe_gpu_dot(count, bottom_data, positive_mask.gpu_data(), statistics.mutable_gpu_data());
    caffe_gpu_dot(count, bottom_square.gpu_data(), positive_mask.gpu_data(), statistics.mutable_gpu_data() + 1);
    caffe_gpu_dot(count, bottom_data, negative_mask.gpu_data(), statistics.mutable_gpu_data() + 2);
    caffe_gpu_dot(count, bottom_square.gpu_data(), negative_mask.gpu_data(), statistics.mutable_gpu_data() + 3);

    const Dtype* statistic_data = statistics.cpu_data();
    top[1]->mutable_cpu_data()[1] = statistic_data[0] / num;
    top[1]->mutable_cpu_data()[2] = (statistic_data[1] - statistic_data[0] / num * statistic_data[0] / num) / num;
    top[1]->mutable_cpu_data()[3] = statistic_data[2] / num;
    top[1]->mutable_cpu_data()[4] = (statistic_data[3] - statistic_data[2] / num * statistic_data[2] / num) / num;
  }

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
