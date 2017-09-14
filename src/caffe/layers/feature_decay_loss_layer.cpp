#include <algorithm>
#include <vector>
#include <math.h>
#include <float.h>

#include "caffe/layers/feature_decay_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void FeatureDecayLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  force_decay_ = this->layer_param_.feature_decay_loss_param().force_decay();
  decay_threshold_ = this->layer_param_.feature_decay_loss_param().decay_threshold();

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
void FeatureDecayLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  sum_multiplier_.Reshape({ bottom[0]->num() });
  caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
  if (top.size() == 2) top[1]->Reshape({ 1 });
}

template <typename Dtype>
void FeatureDecayLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* norm = bottom[0]->cpu_data();
  const int num = bottom[0]->num();
  Dtype* loss = top[0]->mutable_cpu_data();
  Dtype* moving_average_norm = this->blobs_[0]->mutable_cpu_data();

  Dtype mean_norm = caffe_cpu_dot(num, norm, sum_multiplier_.cpu_data());
  mean_norm /= num;
  if (moving_average_norm[0] < 0) {
    moving_average_norm[0] = mean_norm;
  }
  else {
    moving_average_norm[0] = moving_average_norm[0] * 0.99 + mean_norm * 0.01;
  }
  loss[0] = moving_average_norm[0];
}

template <typename Dtype>
void FeatureDecayLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int num = bottom[0]->num();
    const Dtype* loss_weight = top[0]->cpu_diff();
    const Dtype* moving_average_norm = this->blobs_[0]->cpu_data();
    if (force_decay_ || moving_average_norm[0] > decay_threshold_) {
      caffe_set(num, loss_weight[0], bottom[0]->mutable_cpu_diff());
    }
    else {
      caffe_set(num, Dtype(0), bottom[0]->mutable_cpu_diff());
    }
  }
}

INSTANTIATE_CLASS(FeatureDecayLossLayer);
REGISTER_LAYER_CLASS(FeatureDecayLoss);

}  // namespace caffe
