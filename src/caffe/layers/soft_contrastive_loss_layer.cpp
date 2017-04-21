#include <algorithm>
#include <vector>
#include <math.h>
#include <float.h>

#include "caffe/layers/soft_contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void SoftContrastiveLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_GE(bottom.size(), 2);
  positive_margin_ = this->layer_param_.soft_contrastive_loss_param().positive_margin();
  negative_margin_ = this->layer_param_.soft_contrastive_loss_param().negative_margin();
  positive_weight_ = this->layer_param_.soft_contrastive_loss_param().positive_weight();
  negative_weight_ = this->layer_param_.soft_contrastive_loss_param().negative_weight();
  positive_outlier_thresh_ = this->layer_param_.general_contrastive_loss_param().positive_outlier_thresh();
  exponent_scale_ = this->layer_param_.soft_contrastive_loss_param().negative_weight();
  square_ = this->layer_param_.soft_contrastive_loss_param().square();
}

template <typename Dtype>
void SoftContrastiveLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  if (top.size() >= 2) {
    // positive distance, negative distance.
    top[1]->Reshape({ 2 });
  }
  sum_exp_.Reshape({ bottom[0]->num(), 1 });
}

template <typename Dtype>
void SoftContrastiveLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* sum_exp_data = sum_exp_.mutable_cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  Dtype weighted_count = num * (abs(positive_weight_) + (dim - 1)*abs(negative_weight_));
  Dtype positive_distance = Dtype(0);
  Dtype negative_distance = Dtype(0);
  Dtype* loss = top[0]->mutable_cpu_data();

  for (int i = 0; i < num; ++i) {
    sum_exp_data[i] = Dtype(0);
    for (int j = 0; j < dim; ++j) {
      if (j != static_cast<int>(label[i])) {
        if (bottom_data[i * dim + j] < negative_margin_) {
          if (square_) {
            bottom_diff[i * dim + j] = (negative_margin_ - bottom_data[i * dim + j]) * (negative_margin_ - bottom_data[i * dim + j]) * negative_weight_;
            sum_exp_data[i] += exp((negative_margin_ - bottom_data[i * dim + j]) * (negative_margin_ - bottom_data[i * dim + j]) * exponent_scale_);
            negative_distance += negative_margin_ - bottom_data[i * dim + j];
          }
          else {
            bottom_diff[i * dim + j] = (negative_margin_ - bottom_data[i * dim + j]) * negative_weight_;//exp(negative_margin_ - bottom_data[i * dim + j]);
            sum_exp_data[i] += exp((negative_margin_ - bottom_data[i * dim + j]) * exponent_scale_);
            negative_distance += negative_margin_ - bottom_data[i * dim + j];
          }
        }
        else {
          bottom_diff[i * dim + j] = 0;
        }
      }
      else {
        if (square_) {
          Dtype truncated_value = std::min(positive_outlier_thresh_, std::max(Dtype(0), bottom_data[i * dim + j] - positive_margin_));
          positive_distance += truncated_value;
          truncated_value *= truncated_value;
          bottom_diff[i * dim + j] = truncated_value * positive_weight_;
          loss[0] += bottom_diff[i * dim + j];
        }
        else {
          Dtype truncated_value = std::min(positive_outlier_thresh_, std::max(Dtype(0), bottom_data[i * dim + j] - positive_margin_));
          bottom_diff[i * dim + j] = truncated_value * positive_weight_;
          positive_distance += truncated_value;
          loss[0] += bottom_diff[i * dim + j];
        }
      }
    }
    if(sum_exp_data[i] > 0) loss[0] += log(sum_exp_data[i]);
  }
  
  loss[0] /= num;
  if (top.size() >= 2) {
    Dtype* distances = top[1]->mutable_cpu_data();
    distances[0] = positive_distance / num;
    distances[1] = negative_distance / num / (dim - 1);
  }
}

template <typename Dtype>
void SoftContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* sum_exp_data = sum_exp_.cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    Dtype negative_sum = Dtype(0);

    for (int i = 0; i < num; ++i) {
      Dtype min_negative_distance = FLT_MAX;
      int min_negative_index = 0;
      for (int j = 0; j < dim; ++j) {
        if (j == static_cast<int>(label[i])) {
          if (bottom_data[i * dim + j] > positive_margin_ && bottom_data[i * dim + j] < positive_outlier_thresh_) {
            if (square_) {
              bottom_diff[i * dim + j] = bottom_data[i * dim + j] * positive_weight_;
            }
            else {
              bottom_diff[i * dim + j] = positive_weight_;
            }
          }
          else {
            bottom_diff[i * dim + j] = 0;
          }
        }
        else {
          if (bottom_data[i * dim + j] < negative_margin_) {
            if (square_) {
              bottom_diff[i * dim + j] = (bottom_data[i * dim + j] - negative_margin_) * negative_weight_ *
                exp((negative_margin_ - bottom_data[i * dim + j]) * (negative_margin_ - bottom_data[i * dim + j]) * exponent_scale_) / sum_exp_data[i];
            }
            else {
              bottom_diff[i * dim + j] = -negative_weight_ * exp((negative_margin_ - bottom_data[i * dim + j]) * exponent_scale_) / sum_exp_data[i];
            }
          }
          else {
            bottom_diff[i * dim + j] = 0;
          }
        }
      }
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    //Dtype weighted_count = num * (abs(positive_weight_) + (dim - 1)*abs(negative_weight_));
    if (bottom.size() == 3) {
      Dtype weight_sum = Dtype(0);
      for (int i = 0; i < num; ++i) {
        weight_sum += bottom[2]->cpu_data()[i];
      }
      weight_sum /= num;
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < dim; ++j) {
          bottom_diff[i * dim + j] *= bottom[2]->cpu_data()[i] / weight_sum;
        }
      }
    }
    
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(SoftContrastiveLossLayer);
REGISTER_LAYER_CLASS(SoftContrastiveLoss);

}  // namespace caffe
