#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void InsanityForwardTrain(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* slope_data) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] / slope_data[index];
  }
}

// CUDA kernele for forward
template <typename Dtype>
__global__ void InsanityForwardTest(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype slope_data) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] / slope_data;
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void InsanityBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
    const Dtype* slope_data) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) / slope_data[index]);
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void InsanityBackwardTest(const int n, const int channels, const int dim,
                                 const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
                                 const Dtype slope_data) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
                                        + (in_data[index] <= 0) / slope_data);
  }
}

template <typename Dtype>
void InsanityLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0] && lb_ < 0) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  if (this->phase_ == TRAIN) {
    Dtype* slope_data =
        static_cast<Dtype*>(alpha.mutable_gpu_data());
	caffe_gpu_rng_uniform<Dtype>(count, lb_, ub_, slope_data);
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
	InsanityForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, bottom_data, top_data, slope_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    InsanityForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		count, channels, dim, bottom_data, top_data, mean_slope);
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
void InsanityLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0] && lb_ < 0) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* slope_data = alpha.gpu_data();

    if (this->phase_ == TRAIN) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      InsanityBackward<Dtype> << <CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS >> >(
        count, channels, dim, top_diff, bottom_data, bottom_diff, slope_data);
      CUDA_POST_KERNEL_CHECK;
    } else {
      InsanityBackwardTest<Dtype> << <CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS >> >(
        count, channels, dim, top_diff, bottom_data, bottom_diff, mean_slope);
      CUDA_POST_KERNEL_CHECK;
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(InsanityLayer);


}  // namespace caffe
