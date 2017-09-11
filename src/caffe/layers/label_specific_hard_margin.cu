#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/label_specific_hard_margin.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void LabelSpecificHardMarginForward(const int n, const int dim, const Dtype* bottom_data, const Dtype* label,
                                                 const Dtype* row_sum, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      Dtype negative_mean = (row_sum[index] - bottom_data[index * dim + gt]) / (dim - 1);
      top_data[index * dim + gt] = bottom_data[index * dim + gt] / 2 + negative_mean / 2;
    }
  }

  template <typename Dtype>
  __global__ void LabelSpecificHardMarginBackward(const int n, const int dim, const Dtype* top_diff, const Dtype* label,
                                              const Dtype* bottom_data, Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      bottom_diff[index * dim + gt] = top_diff[index * dim + gt] / 2;
    }
  }

  template <typename Dtype>
  void LabelSpecificHardMarginLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* scale_bias = (bottom.size() == 3) ? bottom[2]->cpu_data() : this->blobs_[0]->cpu_data();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    caffe_copy(count, bottom_data, top_data);
    if (this->phase_ == TEST) return;

    caffe_gpu_gemv(CblasNoTrans, num, dim, Dtype(1), bottom_data, sum_multiplier_.gpu_data(), Dtype(0), margins_.mutable_gpu_data());

    // NOLINT_NEXT_LINE(whitespace/operators)
    LabelSpecificHardMarginForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, bottom_data, label_data, margins_.gpu_data(), top_data);
    CUDA_POST_KERNEL_CHECK;

    if (top.size() == 2) {
      top[1]->mutable_cpu_data()[0] = margins_.asum_data();
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
    Dtype* scale_bias_diff = bottom.size() == 3 ? bottom[2]->mutable_cpu_diff() : this->blobs_[0]->mutable_cpu_diff();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    if (propagate_down[0]) {
      caffe_copy(count, top_diff, bottom_diff);
      if (this->phase_ == TEST) return;

      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificHardMarginBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, top_diff, label_data, bottom_data, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    }
  }


  INSTANTIATE_LAYER_GPU_FUNCS(LabelSpecificHardMarginLayer);


}  // namespace caffe
