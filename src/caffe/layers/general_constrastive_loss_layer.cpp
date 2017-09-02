#include <algorithm>
#include <vector>
#include <math.h>
#include <float.h>

#include "caffe/layers/general_contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void GeneralContrastiveLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_GE(bottom.size(), 2);
  positive_margin_ = this->layer_param_.general_contrastive_loss_param().positive_margin();
  negative_margin_ = this->layer_param_.general_contrastive_loss_param().negative_margin();
  positive_weight_ = this->layer_param_.general_contrastive_loss_param().positive_weight();
  negative_weight_ = this->layer_param_.general_contrastive_loss_param().negative_weight();
  need_normalize_negative_ = this->layer_param_.general_contrastive_loss_param().has_normalize_negative();
  negative_gradient_norm_ = this->layer_param_.general_contrastive_loss_param().normalize_negative();
  positive_outlier_thresh_ = this->layer_param_.general_contrastive_loss_param().positive_outlier_thresh();
  max_negative_only_ = this->layer_param_.general_contrastive_loss_param().max_negative_only();
  max_positive_only_ = this->layer_param_.general_contrastive_loss_param().max_positive_only();
  positive_first_ = this->layer_param_.general_contrastive_loss_param().positive_first();
  positive_upper_bound_ = this->layer_param_.general_contrastive_loss_param().positive_upper_bound();
  exp_negative_weight_ = this->layer_param_.general_contrastive_loss_param().exp_negative_weight();
  add_intra_mae_ = this->layer_param_.general_contrastive_loss_param().add_intra_mae();
  max_negative_margin_ = this->layer_param_.general_contrastive_loss_param().max_negative_margin();
  intra_mae_ = Dtype(0);
}

template <typename Dtype>
void GeneralContrastiveLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  if (top.size() >= 2) {
    if (add_intra_mae_) {
      // positive distance, negative distance, intra_mae.
      top[1]->Reshape({ 3 });
    }
    else {
      // positive distance, negative distance.
      top[1]->Reshape({ 2 });
    }
  }
  if (max_negative_only_) {
    max_negative_index_.Reshape({ bottom[0]->num() });
  }
}

template <typename Dtype>
void GeneralContrastiveLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = (bottom.size() == 3) ? bottom[2]->cpu_data() : bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int* max_negative_index_data = NULL;
  if(max_negative_only_ ) max_negative_index_data = max_negative_index_.mutable_cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  Dtype weighted_count = num * (abs(positive_weight_) + (dim - 1)*abs(negative_weight_));
  Dtype positive_distance = Dtype(0);
  Dtype negative_distance = Dtype(0);
  max_positive_index_ = 0;
  if (add_intra_mae_) negative_margin_ = intra_mae_ + this->layer_param_.general_contrastive_loss_param().negative_margin();

  for (int i = 0; i < num; ++i) {
    Dtype same_distance = bottom_data[i * dim + static_cast<int>(label[i])];
    if(max_negative_only_) max_negative_index_data[i] = 0;
    for (int j = 0; j < dim; ++j) {
      if (j == static_cast<int>(label[i])) {
        Dtype truncated_value = std::min(positive_outlier_thresh_, std::max(Dtype(0), bottom_data[i * dim + j] - positive_margin_));
        bottom_diff[i * dim + j] = truncated_value * positive_weight_;
        positive_distance += truncated_value;
        if (max_positive_only_ && bottom_diff[i * dim + j] < bottom_diff[max_positive_index_*dim + static_cast<int>(label[max_positive_index_])]) {
          max_positive_index_ = i;
        }
      }
      else {
        if (bottom_data[i * dim + j] < negative_margin_) {
          if (exp_negative_weight_) {
            bottom_diff[i * dim + j] = exp(-bottom_data[i * dim + j]) * negative_weight_;
          }
          else {
            bottom_diff[i * dim + j] = std::max(
              Dtype(0), negative_margin_ - bottom_data[i * dim + j]) * negative_weight_;
          }
          negative_distance += std::max(
            Dtype(0), negative_margin_ - bottom_data[i * dim + j]);
          if (max_negative_only_ && bottom_diff[i * dim + j] > bottom_diff[i*dim + max_negative_index_data[i]]) {
            max_negative_index_data[i] = j;
          }
        }
        else {
          bottom_diff[i*dim + j] = 0;
        }
      }
    }
    if (positive_first_ && same_distance > positive_upper_bound_) {
      for (int j = 0; j < dim; ++j) {
        if (j != static_cast<int>(label[i])) {
          bottom_diff[i * dim + j] = Dtype(0);
        }
      }
    }
  }
  if (max_positive_only_) {
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; ++j) {
        if (j == static_cast<int>(label[i]) && i != max_positive_index_) {
          bottom_diff[i*dim + j] = 0;
        }
      }
    }
  }
  if (max_negative_only_) {
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; ++j) {
        if (j != static_cast<int>(label[i]) && j != max_negative_index_data[i]) {
          bottom_diff[i*dim + j] = 0;
        }
      }
    }
  }
  intra_mae_ = 0.99 * intra_mae_ + 0.01 * positive_distance / num;
  if (intra_mae_ > max_negative_margin_) intra_mae_ = max_negative_margin_;

  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
  if (top.size() >= 2) {
    Dtype* distances = top[1]->mutable_cpu_data();
    distances[0] = positive_distance / num;
    distances[1] = negative_distance / num / (dim - 1);
    if (add_intra_mae_) {
      distances[2] = intra_mae_;
    }
  }
}

template <typename Dtype>
void GeneralContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = (bottom.size() == 3) ? bottom[2]->cpu_data() : bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    const int* max_negative_index_data = NULL;
    if (max_negative_only_) max_negative_index_data = max_negative_index_.cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    Dtype negative_sum = Dtype(0);

    for (int i = 0; i < num; ++i) {
      Dtype same_distance = bottom_data[i * dim + static_cast<int>(label[i])];
      if (positive_first_ && same_distance > positive_upper_bound_) {
        bottom_diff[i * dim + static_cast<int>(label[i])] = positive_weight_;
        for (int j = 0; j < dim; ++j) {
          if (j != static_cast<int>(label[i])) {
            bottom_diff[i * dim + j] = Dtype(0);
          }
        }
        continue;
      }
      for (int j = 0; j < dim; ++j) {
        if (j == static_cast<int>(label[i])) {
          if (bottom_data[i * dim + j] > positive_margin_ && bottom_data[i * dim + j] < positive_outlier_thresh_
              && !(max_positive_only_ && i != max_positive_index_)) {
            bottom_diff[i * dim + j] = positive_weight_;
          }
          else {
            bottom_diff[i * dim + j] = 0;
          }
        }
        else {
          if (bottom_data[i * dim + j] < negative_margin_) {
            if (max_negative_only_ && j != max_negative_index_data[i]) {
              bottom_diff[i * dim + j] = 0;
            }
            else {
              if (exp_negative_weight_) {
                bottom_diff[i * dim + j] = -1 * exp(-bottom_data[i * dim + j]) * negative_weight_;
                negative_sum += exp(-bottom_data[i * dim + j]) * negative_weight_;
              }
              else {
                bottom_diff[i * dim + j] = -1 * negative_weight_;
                negative_sum += negative_weight_;
              }
              
            }
          }
          else{
            bottom_diff[i * dim + j] = 0;
          }
        }
      }
    }

    if (need_normalize_negative_) {
      negative_sum = abs(negative_sum) / negative_gradient_norm_ / num;

      if (negative_sum > positive_weight_) {
        for (int i = 0; i < num; ++i) {
          for (int j = 0; j < dim; ++j) {
            if (j != static_cast<int>(label[i])) {
              bottom_diff[i * dim + j] /= negative_sum;
            }
          }
        }
      }
    } 

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(GeneralContrastiveLossLayer);
REGISTER_LAYER_CLASS(GeneralContrastiveLoss);

}  // namespace caffe
