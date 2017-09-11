#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/label_specific_hard_margin.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void LabelSpecificHardMarginForward(const int num, const int dim, const Dtype* bottom_data, const Dtype* label,
                                                 Dtype* row_sum, Dtype* top_data, Dtype positive_weight) {
    CUDA_KERNEL_LOOP(index, num) {
      int gt = static_cast<int>(label[index]);
      row_sum[index] = (row_sum[index] - bottom_data[index * dim + gt]) / (dim - 1);
      top_data[index * dim + gt] = bottom_data[index * dim + gt] * positive_weight + row_sum[index] * (1 - positive_weight);
      row_sum[index] = top_data[index * dim + gt] - bottom_data[index * dim + gt];
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificHardMarginBackward(const int num, const int dim, const Dtype* top_diff, const Dtype* label,
                                              const Dtype* bottom_data, Dtype* bottom_diff, Dtype positive_weight) {
    CUDA_KERNEL_LOOP(index, num) {
      int gt = static_cast<int>(label[index]);
      bottom_diff[index * dim + gt] = top_diff[index * dim + gt] * positive_weight;
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificHardMarginBackwardNegative(const int num, const int dim, const Dtype* top_diff, const Dtype* label,
    const Dtype* bottom_data, Dtype* bottom_diff, Dtype positive_weight) {
    CUDA_KERNEL_LOOP(index, num * dim) {
      int n = index / dim;
      int d = index % dim;
      int gt = static_cast<int>(label[n]);
      if (d != gt) {
        bottom_diff[n * dim + d] += top_diff[n * dim + gt] * (1 - positive_weight) / (dim - 1);
      }
    }
  }

  template <typename Dtype>
  void LabelSpecificHardMarginLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    caffe_copy(count, bottom_data, top_data);
    if (this->phase_ == TEST) return;

    caffe_gpu_gemv(CblasNoTrans, num, dim, Dtype(1), bottom_data, sum_multiplier_.gpu_data(), Dtype(0), margins_.mutable_gpu_data());

    // NOLINT_NEXT_LINE(whitespace/operators)
    LabelSpecificHardMarginForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, bottom_data, label_data, margins_.mutable_gpu_data(), top_data, positive_weight);
    CUDA_POST_KERNEL_CHECK;

    if (top.size() == 2) {
      top[1]->mutable_cpu_data()[0] = margins_.asum_data() / Dtype(num) / Dtype(M_PI) * Dtype(180.0);
    }
  }

  template <typename Dtype>
  void LabelSpecificHardMarginLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down,
                                                     const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    if (propagate_down[0]) {
      caffe_copy(count, top_diff, bottom_diff);
      if (this->phase_ == TEST) return;

      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificHardMarginBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, top_diff, label_data, bottom_data, bottom_diff, positive_weight);
      CUDA_POST_KERNEL_CHECK;

      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificHardMarginBackwardNegative<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, top_diff, label_data, bottom_data, bottom_diff, positive_weight);
      CUDA_POST_KERNEL_CHECK;
    }
  }


  INSTANTIATE_LAYER_GPU_FUNCS(LabelSpecificHardMarginLayer);


}  // namespace caffe
