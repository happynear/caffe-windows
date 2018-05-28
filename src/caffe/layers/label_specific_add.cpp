#include <algorithm>
#include <vector>

#include "caffe/layers/label_specific_add_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void LabelSpecificAddLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const LabelSpecificAddParameter& param = this->layer_param_.label_specific_add_param();
    bias_ = param.bias();
    transform_test_ = param.transform_test() & (this->phase_ == TRAIN);
    anneal_bias_ = param.has_bias_base();
    bias_base_ = param.bias_base();
    bias_gamma_ = param.bias_gamma();
    bias_power_ = param.bias_power();
    bias_min_ = param.bias_min();
    bias_max_ = param.bias_max();
    iteration_ = param.iteration();
  }

  template <typename Dtype>
  void LabelSpecificAddLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    if(top[0] != bottom[0]) top[0]->ReshapeLike(*bottom[0]);
    if (top.size() == 2) top[1]->Reshape({ 1 });
  }

template <typename Dtype>
void LabelSpecificAddLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);

  if (!transform_test_ && this->phase_ == TEST) return;

  if (anneal_bias_) {
    bias_ = bias_base_ + pow(((Dtype)1. + bias_gamma_ * iteration_), bias_power_) - (Dtype)1.;
    bias_ = std::max(bias_, bias_min_);
    bias_ = std::min(bias_, bias_max_);
    iteration_++;
  }
  if (top.size() == 2) {
    top[1]->mutable_cpu_data()[0] = bias_;
  }

  for (int i = 0; i < num; ++i) {
    int gt = static_cast<int>(label_data[i]);
    if(top_data[i * dim + gt] > -bias_) top_data[i * dim + gt] += bias_;
  }
}

template <typename Dtype>
void LabelSpecificAddLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom) {
  if (top[0] != bottom[0] && propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int count = bottom[0]->count();
    caffe_copy(count, top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(LabelSpecificAddLayer);
#endif

INSTANTIATE_CLASS(LabelSpecificAddLayer);
REGISTER_LAYER_CLASS(LabelSpecificAdd);

}  // namespace caffe
