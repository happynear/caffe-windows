#include <algorithm>
#include <vector>

#include "caffe/layers/label_specific_rescale_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void LabelSpecificRescaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const LabelSpecificRescaleParameter& param = this->layer_param_.label_specific_rescale_param();
    positive_weight = param.positive_weight();
    negative_weight = param.negative_weight();
    positive_lower_bound = param.positive_lower_bound();//Not implemented
    negative_upper_bound = param.negative_upper_bound();//Not implemented
    rescale_test = param.rescale_test();
    for_ip = param.for_ip();
    positive_weight_base_ = param.positive_weight_base();
    gamma_ = param.gamma();
    power_ = param.power();
    has_positive_weight_min_ = param.has_positive_weight_min();
    positive_weight_min_ = param.positive_weight_min();
    has_positive_weight_max_ = param.has_positive_weight_max();
    positive_weight_max_ = param.positive_weight_max();
    bias_fix_ = param.bias_fix();
    scale_positive_weight_ = param.has_positive_weight_base() & (this->phase_ == TRAIN);
    power_on_positive_ = param.power_on_positive();
    if (power_on_positive_) {
      CHECK(bottom[0] != top[0]) << "Inplace is disabled when using power rescaling.";
    }
    iter_ = param.iteration();
  }

  template <typename Dtype>
  void LabelSpecificRescaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    if (top[0] != bottom[0]) top[0]->ReshapeLike(*bottom[0]);
    if (top.size() == 2) top[1]->Reshape({ 1 });
  }

template <typename Dtype>
void LabelSpecificRescaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  if (scale_positive_weight_) {
    iter_++;
    //lambda_ = base_ * pow(((Dtype)1. + gamma_ * iter_), -power_);
    positive_weight = positive_weight_base_ * pow(((Dtype)1. + gamma_ * iter_), -power_);
    if (has_positive_weight_min_) {
      positive_weight = std::max(positive_weight, positive_weight_min_);
    }
    if (has_positive_weight_max_) {
      positive_weight = std::min(positive_weight, positive_weight_max_);
    }
  }
  if (top.size() >= 2) top[1]->mutable_cpu_data()[0] = positive_weight;

  if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);

  if (!rescale_test && this->phase_ == TEST) return;
  if (positive_weight != Dtype(1.0)) {
    for (int i = 0; i < num; ++i) {
      if ((!for_ip) || top_data[i * dim + static_cast<int>(label_data[i])] > 0) {
        if (power_on_positive_) {
          Dtype bottom_sign = caffe_sign(bottom_data[i * dim + static_cast<int>(label_data[i])]);
          top_data[i * dim + static_cast<int>(label_data[i])] = bottom_sign * pow(abs(bottom_data[i * dim + static_cast<int>(label_data[i])]), positive_weight);
        }
        else {
          top_data[i * dim + static_cast<int>(label_data[i])] *= positive_weight;
          if (bias_fix_) {
            top_data[i * dim + static_cast<int>(label_data[i])] += 1 - positive_weight;
          }
        }
      }
    }
  }
  if (negative_weight != Dtype(1.0)) {
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; j++) {
        if (j != static_cast<int>(label_data[i])) {
          top_data[i * dim + j] *= negative_weight;
        }
      }
    }
  }
}

template <typename Dtype>
void LabelSpecificRescaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                    const vector<bool>& propagate_down,
                                                    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label_data = bottom[1]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    if (top[0] != bottom[0]) caffe_copy(count, top_diff, bottom_diff);
    if (!rescale_test && this->phase_ == TEST) return;

    if (positive_weight != Dtype(1.0)) {
      for (int i = 0; i < num; ++i) {
        if ((!for_ip) || bottom_data[i * dim + static_cast<int>(label_data[i])] > 0) {
          if (power_on_positive_) {
            bottom_diff[i * dim + static_cast<int>(label_data[i])] *= positive_weight * pow(abs(bottom_data[i * dim + static_cast<int>(label_data[i])]), positive_weight - 1);
          }
          else {
            bottom_diff[i * dim + static_cast<int>(label_data[i])] *= positive_weight;
          }
        }
      }
    }
    if (negative_weight != Dtype(1.0)) {
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < dim; j++) {
          if (j != static_cast<int>(label_data[i])) {
            bottom_diff[i * dim + j] *= negative_weight;
          }
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(LabelSpecificRescaleLayer);
#endif

INSTANTIATE_CLASS(LabelSpecificRescaleLayer);
REGISTER_LAYER_CLASS(LabelSpecificRescale);

}  // namespace caffe
