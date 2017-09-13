#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/moving_normalize_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels, const int spatial_dim, Dtype epsilon,
                                const Dtype* data, Dtype* norm_data) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    norm_data[index] = sum + epsilon;
  }
}

template <typename Dtype>
void MovingNormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* square_data = squared_.mutable_gpu_data();
  Dtype* norm_data = norm_.mutable_gpu_data();
  Dtype* moving_average_norm = this->blobs_[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int count = bottom[0]->count();
    Dtype mean_norm = Dtype(0);

  caffe_gpu_powx(num*channels*spatial_dim, bottom_data, Dtype(2), square_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<Dtype> << <CAFFE_GET_BLOCKS(num*spatial_dim),
    CAFFE_CUDA_NUM_THREADS >> > (num, channels, spatial_dim, 1e-12, square_data, norm_data);
  caffe_gpu_powx(num * spatial_dim, norm_data, Dtype(0.5), norm_data);
  caffe_gpu_dot(num, norm_.gpu_data(), sum_multiplier_.gpu_data(), &mean_norm);
  mean_norm /= num * spatial_dim;
  if (moving_average_norm[0] < 0) {
    moving_average_norm[0] = mean_norm;
  }
  else {
    moving_average_norm[0] = moving_average_norm[0] * 0.99 + mean_norm * 0.01;
  }
  if (top.size() == 2) {
    top[1]->mutable_cpu_data()[0] = moving_average_norm[0];
  }
  caffe_gpu_scale(count, Dtype(1) / moving_average_norm[0], bottom_data, top_data);
}



template <typename Dtype>
void MovingNormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* moving_average_norm = this->blobs_[0]->cpu_data();

  int count = bottom[0]->count();
  
  caffe_gpu_scale(count, Dtype(1) / moving_average_norm[0], top_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(MovingNormalizeLayer);


}  // namespace caffe