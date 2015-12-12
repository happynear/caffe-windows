// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/custom_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    Blob<Dtype> mask;
    mask.ReshapeLike(*bottom[0]);
    if (this->layer_param_.noise_param().has_gaussian_std()) {
      caffe_rng_gaussian<Dtype>(count, this->layer_param_.noise_param().bias(), this->layer_param_.noise_param().gaussian_std(), mask.mutable_cpu_data());
    }
    else if (this->layer_param_.noise_param().has_uniform_range()) {
      caffe_rng_uniform<Dtype>(count, this->layer_param_.noise_param().bias() - this->layer_param_.noise_param().uniform_range(),
                        this->layer_param_.noise_param().bias() + this->layer_param_.noise_param().uniform_range(), mask.mutable_cpu_data());
    }
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] + mask.cpu_data()[i];
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(NoiseLayer);
#endif

INSTANTIATE_CLASS(NoiseLayer);
REGISTER_LAYER_CLASS(Noise);

}  // namespace caffe
