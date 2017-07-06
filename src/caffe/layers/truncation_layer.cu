#include <algorithm>
#include <vector>

#include "caffe/layers/truncation_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TruncationForward(const int n, const Dtype* in, Dtype* out,
    Dtype lower_bound, Dtype upper_bound) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = min(max(in[index], lower_bound), upper_bound);
  }
}

template <typename Dtype>
void TruncationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype lower_bound = this->layer_param_.truncation_param().lower_bound();
  Dtype upper_bound = this->layer_param_.truncation_param().upper_bound();
  // NOLINT_NEXT_LINE(whitespace/operators)
  TruncationForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, lower_bound, upper_bound);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void TruncationBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype lower_bound, Dtype upper_bound) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = (in_data[index] > lower_bound && in_data[index] < upper_bound) ? in_diff[index] : 0;
  }
}

template <typename Dtype>
void TruncationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype lower_bound = this->layer_param_.truncation_param().lower_bound();
    Dtype upper_bound = this->layer_param_.truncation_param().upper_bound();
    // NOLINT_NEXT_LINE(whitespace/operators)
    TruncationBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, lower_bound, upper_bound);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(TruncationLayer);


}  // namespace caffe
