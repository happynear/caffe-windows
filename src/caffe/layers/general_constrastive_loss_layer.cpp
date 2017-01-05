#include <algorithm>
#include <vector>

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
}

template <typename Dtype>
void GeneralContrastiveLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  if (top.size() >= 2) {
    // positive distance, negative distance.
    top[1]->Reshape({ 2 });
  }
}

template <typename Dtype>
void GeneralContrastiveLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  Dtype weighted_count = num * (abs(positive_weight_) + (dim - 1)*abs(negative_weight_));
  Dtype positive_distance = Dtype(0);
  Dtype negative_distance = Dtype(0);

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      if (j != static_cast<int>(label[i])) {
        bottom_diff[i * dim + j] = std::max(
          Dtype(0), negative_margin_ - bottom_data[i * dim + j]) * negative_weight_;
        negative_distance += std::max(
          Dtype(0), negative_margin_ - bottom_data[i * dim + j]);
      }
      else {
        bottom_diff[i * dim + j] = std::max(
          Dtype(0), bottom_data[i * dim + j] - positive_margin_) * positive_weight_;
        positive_distance += std::max(
          Dtype(0), bottom_data[i * dim + j] - positive_margin_);
      }
    }
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
  if (top.size() >= 2) {
    Dtype* distances = top[1]->mutable_cpu_data();
    distances[0] = positive_distance / num;
    distances[1] = negative_distance / num / (dim - 1);
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
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    Dtype negative_sum = Dtype(0);

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; ++j) {
        if (j == static_cast<int>(label[i])) {
          if (bottom_data[i * dim + j] > positive_margin_) {
            bottom_diff[i * dim + j] = positive_weight_;
          }
          else {
            bottom_diff[i * dim + j] = 0;
          }
        }
        else {
          if (bottom_data[i * dim + j] < negative_margin_) {
            Dtype distance_fix = 1;// std::min(Dtype(1), (negative_margin_ - bottom_data[i * dim + j]) / (bottom_data[i * dim + j] + Dtype(1e-6)));
            bottom_diff[i * dim + j] = -negative_weight_ * distance_fix;
            negative_sum += negative_weight_ * distance_fix;
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

INSTANTIATE_CLASS(GeneralContrastiveLossLayer);
REGISTER_LAYER_CLASS(GeneralContrastiveLoss);

}  // namespace caffe
