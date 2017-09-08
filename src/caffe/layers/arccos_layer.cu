#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/arccos_layer.hpp"

namespace caffe {

  // CUDA kernele for forward
  template <typename Dtype>
  __global__ void ArccosForward(const int n, const Dtype* in, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      Dtype fixed_in_data = min(in[index], Dtype(1.0) - Dtype(1e-4));
      fixed_in_data = max(fixed_in_data, Dtype(-1.0) + Dtype(1e-4));
      out[index] = acosf(fixed_in_data);
    }
  }

  // CUDA kernel for bottom backward
  template <typename Dtype>
  __global__ void ArccosBackward(const int n, const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff) {
    CUDA_KERNEL_LOOP(index, n) {
      Dtype fixed_in_data = min(in_data[index], Dtype(1.0) - Dtype(1e-4));
      fixed_in_data = max(fixed_in_data, Dtype(-1.0) + Dtype(1e-4));
      out_diff[index] = in_diff[index] * -1 / sqrtf(1.0f - fixed_in_data * fixed_in_data);
    }
  }

  template <typename Dtype>
  void ArccosLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();

    // NOLINT_NEXT_LINE(whitespace/operators)
    ArccosForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      count, bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  void ArccosLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const int count = bottom[0]->count();

    // Propagate to bottom
    if (propagate_down[0]) {
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      // NOLINT_NEXT_LINE(whitespace/operators)
      ArccosBackward<Dtype> << <CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS >> >(
          count, top_diff, bottom_data, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    }
  }


  INSTANTIATE_LAYER_GPU_FUNCS(ArccosLayer);


}  // namespace caffe
