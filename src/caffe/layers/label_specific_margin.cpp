#include <algorithm>
#include <vector>

#include "caffe/layers/label_specific_margin_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void LabelSpecificMarginLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const LabelSpecificMarginParameter& param = this->layer_param_.label_specific_margin_param();
    has_margin_base_ = param.has_margin_base();
    margin_base_ = param.margin_base();
    has_margin_max_ = param.has_margin_max();
    margin_max_ = param.margin_max();
    gamma_ = param.gamma();
    power_ = param.power();
    iter_ = param.iteration();
    margin_on_test_ = param.margin_on_test() & (this->phase_ == TRAIN);
    auto_tune_ = param.auto_tune();
    type_ = param.type();
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    }
    else {
      this->blobs_.resize(1);
      if (auto_tune_) {
        this->blobs_[0].reset(new Blob<Dtype>({ 3 }));
        caffe_set(5, Dtype(0), this->blobs_[0]->mutable_cpu_data());
      }
      else {
        this->blobs_[0].reset(new Blob<Dtype>({ 1 }));
      }
      this->blobs_[0]->mutable_cpu_data()[0] = margin_base_;
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
  void LabelSpecificMarginLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
    if (top.size() == 2) {
      if (auto_tune_ && bottom.size() < 3) {
        top[1]->Reshape({ 3 });
        positive_mask.ReshapeLike(*bottom[0]);
        negative_mask.ReshapeLike(*bottom[0]);
        bottom_angle.ReshapeLike(*bottom[0]);
        //bottom_square.ReshapeLike(*bottom[0]);
      }
      else {
        top[1]->Reshape({ 1 });
      }
    }
    if (type_ == LabelSpecificMarginParameter_MarginType::LabelSpecificMarginParameter_MarginType_SOFT) {
      theta.ReshapeLike(*bottom[0]);
    }
    if (bottom.size() == 3) {
      positive_data.Reshape({ bottom[0]->num() });
      if (sum_multiplier_.Reshape({ bottom[0]->num() })) {
        caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
      }
      
    }
    
  }

template <typename Dtype>
void LabelSpecificMarginLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* margin = this->blobs_[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  if (has_margin_base_) {
    margin[0] = margin_base_ + pow(((Dtype)1. + gamma_ * iter_), power_) - 1;
    iter_++;
  }
  if (has_margin_max_) {
    margin[0] = std::min(margin[0], margin_max_);
  }
  
  caffe_copy(count, bottom_data, top_data);
  if (top.size() >= 2) top[1]->mutable_cpu_data()[0] = margin[0];

  if (!margin_on_test_ && this->phase_ == TEST) return;
  if (margin[0] != Dtype(0.0)) {
    for (int i = 0; i < num; ++i) {
      int gt = static_cast<int>(label_data[i]);
      top_data[i * dim + gt] = bottom_data[i * dim + gt] * cosf(margin[0] / 180 * M_PI) -
        sqrt(1 - bottom_data[i * dim + gt] * bottom_data[i * dim + gt] + 1e-12) * sinf(margin[0] / 180 * M_PI);
    }
  }
}

template <typename Dtype>
void LabelSpecificMarginLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                    const vector<bool>& propagate_down,
                                                    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label_data = bottom[1]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* margin = this->blobs_[0]->mutable_cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    caffe_copy(count, top_diff, bottom_diff);
    if (!margin_on_test_ && this->phase_ == TEST) return;

    if (margin[0] != Dtype(0.0)) {
      for (int i = 0; i < num; ++i) {
        int gt = static_cast<int>(label_data[i]);
        bottom_diff[i * dim + gt] = top_diff[i * dim + gt] * (cosf(margin[0] / 180 * M_PI) -
                                                              bottom_data[i * dim + gt] / sqrt(1 - bottom_data[i * dim + gt] * bottom_data[i * dim + gt] + 1e-12)
                                                              * sinf(margin[0] / 180 * M_PI));
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(LabelSpecificMarginLayer);
#endif

INSTANTIATE_CLASS(LabelSpecificMarginLayer);
REGISTER_LAYER_CLASS(LabelSpecificMargin);

}  // namespace caffe
