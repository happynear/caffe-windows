#include <algorithm>
#include <vector>

#include "caffe/layers/soft_truncation_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftTruncationForward(const int n, const Dtype* in, Dtype* out,
    Dtype c) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 1 - exp(in[index] / (-c));
  }
}

template <typename Dtype>
void SoftTruncationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype c = this->layer_param_.soft_truncation_param().c();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftTruncationForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, c);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SoftTruncationBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype c) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (1 - in_data[index]) / c;
  }
}

template <typename Dtype>
void SoftTruncationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype c = this->layer_param_.soft_truncation_param().c();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftTruncationBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_diff, c);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SoftTruncationLayer);


}  // namespace caffe
