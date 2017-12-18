#include <vector>

#include "caffe/custom_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    Blob<Dtype> mask;
    mask.ReshapeLike(*bottom[0]);
    if (this->layer_param_.noise_param().has_gaussian_std()) {
      caffe_gpu_rng_gaussian<Dtype>(count, (Dtype)this->layer_param_.noise_param().bias(), (Dtype)this->layer_param_.noise_param().gaussian_std(), mask.mutable_gpu_data());
    }
    else if (this->layer_param_.noise_param().has_uniform_range()) {
      caffe_gpu_rng_uniform<Dtype>(count, (Dtype)this->layer_param_.noise_param().bias() - (Dtype)this->layer_param_.noise_param().uniform_range(),
                                   (Dtype)this->layer_param_.noise_param().bias() + (Dtype)this->layer_param_.noise_param().uniform_range(), mask.mutable_gpu_data());
    }
    caffe_gpu_add(count, bottom_data, mask.gpu_data(), top_data);
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NoiseLayer);


}  // namespace caffe
