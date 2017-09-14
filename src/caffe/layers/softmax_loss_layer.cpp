#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  normalize_type_ =
    this->layer_param_.softmax_param().normalize_type();
  if (normalize_type_ == "Softmax") {
    LayerParameter softmax_param(this->layer_param_);
    softmax_param.set_type("Softmax");
    softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
    softmax_bottom_vec_.clear();
    softmax_bottom_vec_.push_back(bottom[0]);
    softmax_top_vec_.clear();
    softmax_top_vec_.push_back(&prob_);
    softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  }
  else if(normalize_type_ == "L2" || normalize_type_ == "L1") {
    LayerParameter normalize_param(this->layer_param_);
    normalize_param.set_type("Normalize");
    softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(normalize_param);
    softmax_bottom_vec_.clear();
    softmax_bottom_vec_.push_back(bottom[0]);
    softmax_top_vec_.clear();
    softmax_top_vec_.push_back(&prob_);
    softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  }
  else {
    NOT_IMPLEMENTED;
  }

  label_smooth_factor_ = this->layer_param_.softmax_param().label_smooth_factor();
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  has_hard_ratio_ =
    this->layer_param_.softmax_param().has_hard_ratio();
  if (has_hard_ratio_) {
    hard_ratio_ = this->layer_param_.softmax_param().hard_ratio();
    CHECK_GE(hard_ratio_, 0);
    CHECK_LE(hard_ratio_, 1);
  }
  has_cutting_point_ =
    this->layer_param_.softmax_param().has_cutting_point();
  if (has_cutting_point_) {
    cutting_point_ = this->layer_param_.softmax_param().cutting_point();
    CHECK_GE(cutting_point_, 0);
    CHECK_LE(cutting_point_, 1);
  }
  has_hard_mining_label_ = this->layer_param_.softmax_param().has_hard_mining_label();
  if (has_hard_mining_label_) {
    hard_mining_label_ = this->layer_param_.softmax_param().hard_mining_label();
  }
  has_class_weight_ = (this->layer_param_.softmax_param().class_weight_size() != 0);
  softmax_axis_ =
    bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  if (has_class_weight_) {
    class_weight_.Reshape({ bottom[0]->shape(softmax_axis_) });
    CHECK_EQ(this->layer_param_.softmax_param().class_weight().size(), bottom[0]->shape(softmax_axis_));
    for (int i = 0; i < bottom[0]->shape(softmax_axis_); i++) {
      class_weight_.mutable_cpu_data()[i] = (Dtype)this->layer_param_.softmax_param().class_weight(i);
    }
  }
  else {
    if (bottom.size() == 3) {
      class_weight_.Reshape({ bottom[0]->shape(softmax_axis_) });
      for (int i = 0; i < bottom[0]->shape(softmax_axis_); i++) {
        class_weight_.mutable_cpu_data()[i] = (Dtype)1.0;
      }
    }
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  counts_.Reshape({ outer_num_, inner_num_ });
  loss_.Reshape({ outer_num_, inner_num_ });
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (bottom.size() == 3) {
    CHECK_EQ(outer_num_ * inner_num_, bottom[2]->count())
      << "Number of loss weights must match number of label.";
  }
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  if (has_class_weight_) {
    CHECK_EQ(class_weight_.count(), bottom[0]->shape(1));
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, Dtype valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = valid_count;
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  Dtype count = 0;
  Dtype loss = 0;
  if (bottom.size() == 2) {
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; j++) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, prob_.shape(softmax_axis_));
        loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
          Dtype(FLT_MIN)));
        count += 1;
      }
    }
  }
  else if(bottom.size() == 3) {
    const Dtype* weights = bottom[2]->cpu_data();
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; j++) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        const Dtype weight_value = weights[i * inner_num_ + j] * (has_class_weight_? class_weight_.cpu_data()[label_value] : 1.0);
        if (weight_value == 0) continue;
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, prob_.shape(softmax_axis_));
        loss -= weight_value * log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
          Dtype(FLT_MIN)));
        count += weight_value;
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    Dtype count = 0;
    if (bottom.size() == 2) {
      for (int i = 0; i < outer_num_; ++i) {
        for (int j = 0; j < inner_num_; ++j) {
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] = 0;
            }
          }
          else {
            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
            count += 1;
          }
        }
      }
    }
    else if (bottom.size() == 3) {
      const Dtype* weights = bottom[2]->cpu_data();
      for (int i = 0; i < outer_num_; ++i) {
        for (int j = 0; j < inner_num_; ++j) {
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);
          const Dtype weight_value = weights[i * inner_num_ + j];
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] = 0;
            }
          }
          else {
            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] *= weight_value * (has_class_weight_ ? class_weight_.cpu_data()[label_value] : 1.0);
            }
            if(weight_value != 0) count += weight_value;
          }
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
