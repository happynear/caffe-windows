/*
Based on https://github.com/zimenglan-sysu-512/Focal-Loss.
Paper https://arxiv.org/abs/1708.02002.
NOTICE: This layer is NOT the original focal loss layer.
I changed 1-p to 1+p for face verification. If you want the original version, 
replace the cpp and cu files from https://github.com/zimenglan-sysu-512/Focal-Loss.
*/
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void FocalLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // softmax laye setup
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    LayerParameter softmax_param(this->layer_param_);
    softmax_param.set_type("Softmax");
    softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
    softmax_bottom_vec_.clear();
    softmax_bottom_vec_.push_back(bottom[0]);
    softmax_top_vec_.clear();
    softmax_top_vec_.push_back(&prob_);
    softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

    // ignore label
    has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
    if (has_ignore_label_) {
      ignore_label_ = this->layer_param_.loss_param().ignore_label();
    }

    // normalization
    if (!this->layer_param_.loss_param().has_normalization() &&
        this->layer_param_.loss_param().has_normalize()) {
      normalization_ = this->layer_param_.loss_param().normalize() ?
        LossParameter_NormalizationMode_VALID :
        LossParameter_NormalizationMode_BATCH_SIZE;
    }
    else {
      normalization_ = this->layer_param_.loss_param().normalization();
    }

    // focal loss parameter
    FocalLossParameter focal_loss_param = this->layer_param_.focal_loss_param();
    alpha_ = focal_loss_param.alpha();
    beta_ = focal_loss_param.beta();
    gamma_ = focal_loss_param.gamma();
    type_ = focal_loss_param.type();
    LOG(INFO) << "alpha: " << alpha_;
    LOG(INFO) << "beta: " << beta_;
    LOG(INFO) << "gamma: " << gamma_;
    LOG(INFO) << "type: " << type_;
    CHECK_GE(gamma_, 0) << "gamma must be larger than or equal to zero";
    CHECK_GT(alpha_, 0) << "alpha must be larger than zero";
    // CHECK_LE(alpha_, 1) << "alpha must be smaller than or equal to one";
  }

  template <typename Dtype>
  void FocalLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // softmax laye reshape
    LossLayer<Dtype>::Reshape(bottom, top);
    softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);

    // cross-channels
    softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
    outer_num_ = bottom[0]->count(0, softmax_axis_);
    inner_num_ = bottom[0]->count(softmax_axis_ + 1);
    CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";

    // softmax output
    if (top.size() >= 2) {
      top[1]->ReshapeLike(*bottom[0]);
    }

    // log(p_t)
    log_prob_.ReshapeLike(*bottom[0]);
    CHECK_EQ(prob_.count(), log_prob_.count());
    // (1 + p_t) ^ gamma
    power_prob_.ReshapeLike(*bottom[0]);
    CHECK_EQ(prob_.count(), power_prob_.count());
    // 1
    ones_.ReshapeLike(*bottom[0]);
    CHECK_EQ(prob_.count(), ones_.count());
    caffe_set(prob_.count(), Dtype(alpha_), ones_.mutable_cpu_data());
  }

  template <typename Dtype>
  Dtype FocalLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
    Dtype normalizer;
    switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      }
      else {
        normalizer = Dtype(valid_count);
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
  void FocalLossLayer<Dtype>::compute_intermediate_values_of_cpu() {
    // compute the corresponding variables
    const int count = prob_.count();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* ones_data = ones_.cpu_data();
    Dtype* log_prob_data = log_prob_.mutable_cpu_data();
    Dtype* power_prob_data = power_prob_.mutable_cpu_data();

    /// log(p_t)
    const Dtype eps = Dtype(FLT_MIN); // where FLT_MIN = 1.17549e-38, here u can change it
                                      // more stable
    for (int i = 0; i < prob_.count(); i++) {
      log_prob_data[i] = log(std::max(prob_data[i], eps));
    }
    /// caffe_log(count,  prob_data, log_prob_data);

    if (type_ == FocalLossParameter::ONEADDP) {
      /// (1 + p_t) ^ gamma
      caffe_add(count, ones_data, prob_data, power_prob_data);
    }
    else {
      /// (1 - p_t) ^ gamma
      caffe_sub(count, ones_data, prob_data, power_prob_data);
    }
    caffe_powx(count, power_prob_.cpu_data(), gamma_, power_prob_data);
    //caffe_scal(count, alpha_, power_prob_data);
  }

  template <typename Dtype>
  void FocalLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // The forward pass computes the softmax prob values.
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

    // compute all needed values
    compute_intermediate_values_of_cpu();
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* log_prob_data = log_prob_.cpu_data();
    const Dtype* power_prob_data = power_prob_.cpu_data();

    // compute loss
    int count = 0;
    int channels = prob_.shape(softmax_axis_);
    int dim = prob_.count() / outer_num_;

    Dtype loss = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; j++) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, channels);
        const int index = i * dim + label_value * inner_num_ + j;
        // FL(p_t) = -(1 + p_t) ^ gamma * log(p_t)
        // loss -= std::max(power_prob_data[index] * log_prob_data[index],
        //                      Dtype(log(Dtype(FLT_MIN))));
        loss -= power_prob_data[index] * log_prob_data[index];
        ++count;
      }
    }

    // prob
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
    if (top.size() == 2) {
      top[1]->ShareData(prob_);
    }
  }

  template <typename Dtype>
  void FocalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }

    if (propagate_down[0]) {
      // data
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      const Dtype* prob_data = prob_.cpu_data();
      const Dtype* label = bottom[1]->cpu_data();
      // intermidiate  
      const Dtype* log_prob_data = log_prob_.cpu_data();
      const Dtype* power_prob_data = power_prob_.cpu_data();

      int count = 0;
      int channels = bottom[0]->shape(softmax_axis_);
      int dim = prob_.count() / outer_num_;
      const Dtype eps = 1e-10;

      for (int i = 0; i < outer_num_; ++i) {
        for (int j = 0; j < inner_num_; ++j) {
          // label
          const int label_value = static_cast<int>(label[i * inner_num_ + j]);

          // ignore label
          if (has_ignore_label_ && label_value == ignore_label_) {
            for (int c = 0; c < channels; ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] = 0;
            }
            continue;
          }

          // the gradient from FL w.r.t p_t, here ignore the `sign`
          int ind_i = i * dim + label_value * inner_num_ + j; // index of ground-truth label
          Dtype grad;
          if (type_ == FocalLossParameter::ONEADDP) {
            grad = gamma_ * (power_prob_data[ind_i] / std::max(1 + prob_data[ind_i], eps))
              * log_prob_data[ind_i] * prob_data[ind_i]
              + power_prob_data[ind_i];
          }
          else {
            grad = -gamma_ * (power_prob_data[ind_i] / std::max(1 - prob_data[ind_i], eps))
              * log_prob_data[ind_i] * prob_data[ind_i]
              + power_prob_data[ind_i];
          }
          
          // the gradient w.r.t input data x
          for (int c = 0; c < channels; ++c) {
            int ind_j = i * dim + c * inner_num_ + j;
            if (c == label_value) {
              CHECK_EQ(ind_i, ind_j);
              // if i == j, (here i,j are refered for derivative of softmax)
              bottom_diff[ind_j] = grad * (prob_data[ind_i] - 1);
            }
            else {
              // if i != j, (here i,j are refered for derivative of softmax)
              bottom_diff[ind_j] = grad * prob_data[ind_j];
            }
          }
          // count                    
          ++count;
        }
      }
      // Scale gradient
      Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, count);
      caffe_scal(prob_.count(), loss_weight, bottom_diff);
    }
  }

#ifdef CPU_ONLY
  STUB_GPU(FocalLossLayer);
#endif

  INSTANTIATE_CLASS(FocalLossLayer);
  REGISTER_LAYER_CLASS(FocalLoss);

}  // namespace caffe
