#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_sum_dim1(const int num, const int dim, Dtype epsilon,
                                const Dtype* data, Dtype* norm_data) {
  CUDA_KERNEL_LOOP(index, num) {
    Dtype sum = 0;
    for (int d = 0; d < dim; ++d) {
      sum += data[index * dim + d];
    }
    norm_data[index] = sum + epsilon;
  }
}

template <typename Dtype>
__global__ void kernel_channel_scale(const int num, const int dim,
                                     const Dtype* data, const Dtype* norm_data,
                                     Dtype* output_data) {
  CUDA_KERNEL_LOOP(index, num * dim) {
    int n = index / dim;
    output_data[index] = data[index] * norm_data[n];
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int dim,
                                   const Dtype* data_1, const Dtype* data_2,
                                   Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num) {
    Dtype dot = 0;
    for (int d = 0; d < dim; ++d) {
      dot += data_1[index * dim + d] * data_2[index * dim + d];
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* square_data = squared_.mutable_gpu_data();
  Dtype* norm_data = norm_.mutable_gpu_data();
  Dtype normsqr;
  int n = bottom[0]->num();
  int d = bottom[0]->count() / n;
  if (normalize_type_ == "L2") {
    caffe_gpu_powx(n*d, bottom_data, Dtype(2), square_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_sum_dim1<Dtype> << <CAFFE_GET_BLOCKS(n),
      CAFFE_CUDA_NUM_THREADS >> >(n, d, 1e-6, square_data, norm_data);
    caffe_gpu_powx(n, norm_data, Dtype(-0.5), norm_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_scale<Dtype> << <CAFFE_GET_BLOCKS(n*d),
      CAFFE_CUDA_NUM_THREADS >> >(n, d, bottom_data, norm_data, top_data);
  }
  else if (normalize_type_ == "L1") {
    caffe_gpu_abs(n*d, bottom_data, square_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_sum_dim1<Dtype> << <CAFFE_GET_BLOCKS(n),
      CAFFE_CUDA_NUM_THREADS >> >(n, d, 1e-6, square_data, norm_data);
    caffe_gpu_powx(n, norm_data, Dtype(-1), norm_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_scale<Dtype> << <CAFFE_GET_BLOCKS(n*d),
      CAFFE_CUDA_NUM_THREADS >> >(n, d, bottom_data, norm_data, top_data);
  }
  else {
    NOT_IMPLEMENTED;
  }
}



template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* square_data = squared_.gpu_data();
  const Dtype* norm_data = norm_.gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* norm_diff = norm_.mutable_gpu_data();

  int n = top[0]->num();
  int d = top[0]->count() / n;
  Dtype a;
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_dot<Dtype> << <CAFFE_GET_BLOCKS(n),
    CAFFE_CUDA_NUM_THREADS >> >(n, d, top_data, top_diff, norm_diff);

  if (normalize_type_ == "L2") {
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_scale<Dtype> << <CAFFE_GET_BLOCKS(n*d),
      CAFFE_CUDA_NUM_THREADS >> >(n, d, top_data, norm_diff, bottom_diff);
  }
  else if (normalize_type_ == "L1") {
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_scale<Dtype> << <CAFFE_GET_BLOCKS(n*d),
      CAFFE_CUDA_NUM_THREADS >> >(n, d, square_data, norm_diff, bottom_diff);
  }
  else {
    NOT_IMPLEMENTED;
  }

  caffe_gpu_sub(n * d, top_diff, bottom_diff, bottom_diff);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_scale<Dtype> << <CAFFE_GET_BLOCKS(n*d),
    CAFFE_CUDA_NUM_THREADS >> >(n, d, bottom_diff, norm_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);


}  // namespace caffe