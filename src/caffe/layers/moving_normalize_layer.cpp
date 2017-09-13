#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/moving_normalize_layer.hpp"

namespace caffe {

#define sign(x) (Dtype(0) < (x)) - ((x) < Dtype(0))

template <typename Dtype>
void MovingNormalizeLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  }
  else {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>({ 1 }));
    this->blobs_[0]->mutable_cpu_data()[0] = -1;
  }
  if (this->layer_param_.param_size() == 0) {
    ParamSpec* fixed_param_spec = this->layer_param_.add_param();
    fixed_param_spec->set_lr_mult(0.f);
    fixed_param_spec->set_decay_mult(0.f);
  }
  else {
    CHECK_EQ(this->layer_param_.param(0).lr_mult(), 0.f)
      << "Cannot configure statistics as layer parameters.";
  }
}

template <typename Dtype>
void MovingNormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  squared_.ReshapeLike(*bottom[0]);
  if (top.size() == 2) {
    top[1]->Reshape({ 1 });
  }
  norm_.Reshape(bottom[0]->num(), 1,
                bottom[0]->height(), bottom[0]->width());
  sum_multiplier_.Reshape(bottom[0]->num(), 1,
                          bottom[0]->height(), bottom[0]->width());
  caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void MovingNormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MovingNormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(MovingNormalizeLayer);
#endif

INSTANTIATE_CLASS(MovingNormalizeLayer);
REGISTER_LAYER_CLASS(MovingNormalize);

}  // namespace caffe