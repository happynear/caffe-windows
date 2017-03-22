#include <vector>

#include "caffe/layers/infimum_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void LabelSpecificRescalePositive(const int n, const Dtype* in, Dtype* loss, Dtype infimum) {
    CUDA_KERNEL_LOOP(index, n) {
      out[index * dim + static_cast<int>(label[index])] = in[index * dim + static_cast<int>(label[index])] * positive_weight;
    }
  }

template <typename Dtype>
void InfimumLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();

  Dtype* loss = top[0]->mutable_cpu_data();
  caffe_gpu_dot(bottom[0]->count(), bottom_data, label, loss);
  loss[0] /= bottom[0]->count();
}

template <typename Dtype>
void InfimumLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype loss_weight = top[0]->cpu_diff()[0] / bottom[0]->count();
  if (propagate_down[0]) {
    caffe_gpu_scale(bottom[0]->count(), loss_weight, bottom[1]->gpu_data(), bottom[0]->mutable_gpu_data());
  }
  if (propagate_down[1]) {
    caffe_gpu_scale(bottom[1]->count(), loss_weight, bottom[0]->gpu_data(), bottom[1]->mutable_gpu_data());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InfimumLossLayer);

}  // namespace caffe
