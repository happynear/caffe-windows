#include <algorithm>
#include <vector>

#include "caffe/layers/label_specific_affine_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void LabelSpecificAffineLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const LabelSpecificAffineParameter& param = this->layer_param_.label_specific_affine_param();
    scale_base_ = param.scale_base();
    scale_gamma_ = param.scale_gamma();
    scale_power_ = param.scale_power();
    scale_max_ = param.scale_max();
    bias_base_ = param.bias_base();
    bias_gamma_ = param.bias_gamma();
    bias_power_ = param.bias_power();
    bias_max_ = param.bias_max();
    power_base_ = param.power_base();
    power_gamma_ = param.power_gamma();
    power_power_ = param.power_power();
    power_min_ = param.power_min();
    iteration_ = param.iteration();
    reset_ = param.reset();
    transform_test_ = param.transform_test() & (this->phase_ == TRAIN);
    auto_tune_ = param.auto_tune();//doesn't work.
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
      if (reset_) {
        this->blobs_[0]->mutable_cpu_data()[0] = scale_base_;
        this->blobs_[0]->mutable_cpu_data()[1] = bias_base_;
      }
    }
    else {
      this->blobs_.resize(1);
      this->blobs_[0].reset(new Blob<Dtype>({ 3 }));
      this->blobs_[0]->mutable_cpu_data()[0] = scale_base_;
      this->blobs_[0]->mutable_cpu_data()[1] = bias_base_;
    }
  }

  template <typename Dtype>
  void LabelSpecificAffineLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
    if (top.size() == 2) {
      top[1]->Reshape({ 3 });
    }
    if (auto_tune_ || bottom.size() == 3) {
      selected_value_.Reshape({ bottom[0]->num() });
      sum_multiplier_.Reshape({ bottom[0]->num() });
      caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
    }
  }

template <typename Dtype>
void LabelSpecificAffineLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* scale_bias = (bottom.size() == 3) ? bottom[2]->cpu_data() : this->blobs_[0]->cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  if (this->phase_ == TEST) {
    scale = Dtype(1);
    bias = Dtype(0);
    power = Dtype(1);
  }
  else {
    if (auto_tune_) {
      scale = scale_bias[0];
      bias = scale_bias[1];
      power = scale_bias[2];
    }
    else {
      scale = scale_base_ * pow(((Dtype)1. + scale_gamma_ * iteration_), scale_power_);
      bias = bias_base_ + pow(((Dtype)1. + bias_gamma_ * iteration_), bias_power_) - (Dtype)1.;
      power = power_base_ * pow(((Dtype)1. + power_gamma_ * iteration_), power_power_);
      scale = std::min(scale, scale_max_);
      bias = std::min(bias, bias_max_);
      power = std::max(power, power_min_);
      iteration_++;
    }
  }

  caffe_copy(count, bottom_data, top_data);
  if (top.size() >= 2) {
    top[1]->mutable_cpu_data()[0] = scale;
    top[1]->mutable_cpu_data()[1] = bias;
    top[1]->mutable_cpu_data()[2] = power;
  }

  if (!transform_test_ && this->phase_ == TEST) return;
  for (int i = 0; i < num; ++i) {
    int gt = static_cast<int>(label_data[i]);
    top_data[i * dim + gt] = pow(bottom_data[i * dim + gt], power) * scale * pow(Dtype(M_PI), Dtype(1) - power) + bias;
    if (top_data[i * dim + gt] > M_PI - 1e-4) top_data[i * dim + gt] = M_PI - 1e-4;
  }
}

template <typename Dtype>
void LabelSpecificAffineLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_bias_diff = bottom.size() == 3? bottom[2]->mutable_cpu_diff() : this->blobs_[0]->mutable_cpu_diff();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  if (propagate_down[0]) {
    caffe_copy(count, top_diff, bottom_diff);
    if (!transform_test_ && this->phase_ == TEST) return;

    for (int i = 0; i < num; ++i) {
      int gt = static_cast<int>(label_data[i]);
      bottom_diff[i * dim + gt] = top_diff[i * dim + gt] * scale * pow(Dtype(M_PI), Dtype(1) - power) * power * pow(bottom_data[i * dim + gt], power - 1);
    }
  }

  if (auto_tune_ || (bottom.size() == 3 && propagate_down[2])) {
    scale_bias_diff[0] = Dtype(0);
    scale_bias_diff[0] = Dtype(1);
    for (int i = 0; i < num; ++i) {
      int gt = static_cast<int>(label_data[i]);
      scale_bias_diff[0] += top_diff[i * dim + gt] * bottom_data[i * dim + gt];
      scale_bias_diff[1] += top_diff[i * dim + gt];
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(LabelSpecificAffineLayer);
#endif

INSTANTIATE_CLASS(LabelSpecificAffineLayer);
REGISTER_LAYER_CLASS(LabelSpecificAffine);

}  // namespace caffe
