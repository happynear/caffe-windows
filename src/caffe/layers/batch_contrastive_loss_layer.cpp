#include <algorithm>
#include <vector>
#include <math.h>
#include <float.h>

#include "caffe/layers/batch_contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void BatchContrastiveLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_GE(bottom.size(), 2);
  positive_margin_ = this->layer_param_.batch_contrastive_loss_param().positive_margin();
  negative_margin_ = this->layer_param_.batch_contrastive_loss_param().negative_margin();
  positive_weight_ = this->layer_param_.batch_contrastive_loss_param().positive_weight();
  negative_weight_ = this->layer_param_.batch_contrastive_loss_param().negative_weight();
  max_only_ = this->layer_param_.batch_contrastive_loss_param().max_only();
}

template <typename Dtype>
void BatchContrastiveLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[0]->channels());
  if (top.size() >= 2) {
    // positive distance, negative distance.
    top[1]->Reshape({ 2 });
  }
}

template <typename Dtype>
void BatchContrastiveLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  Dtype positive_distance = Dtype(0);
  int positive_count = 0;
  Dtype negative_distance = Dtype(0);
  int negative_count = 0;
  max_positive_1_ = -1; max_positive_2_ = -1;
  min_negative_1_ = -1; min_negative_2_ = -1;
  Dtype max_positive_value = -FLT_MAX;
  Dtype min_negative_value = FLT_MAX;
  Dtype* loss = top[0]->mutable_cpu_data();

  for (int i = 0; i < num; ++i) {
    for (int j = i + 1; j < num; ++j) {
      if (label[i] == label[j]) {
        positive_distance += bottom_data[i*num + j];
        positive_count++;
        if (bottom_data[i*num + j] > positive_margin_) {
          if (max_only_ && bottom_data[i*num + j] > max_positive_value) {
            max_positive_value = bottom_data[i*num + j];
            max_positive_1_ = i;
            max_positive_2_ = j;
          }
          loss[0] += positive_weight_ * (bottom_data[i*num + j] - positive_margin_);
        }
      }
      else {
        negative_distance += bottom_data[i*num + j];
        negative_count++;
        if (bottom_data[i*num + j] < negative_margin_) {
          if (max_only_ && bottom_data[i*num + j] < min_negative_value) {
            min_negative_value = bottom_data[i*num + j];
            min_negative_1_ = i;
            min_negative_2_ = j;
          }
          loss[0] += negative_weight_ * (negative_margin_ - bottom_data[i*num + j]);
        }
      }
    }
  }
  if (max_only_) {
    loss[0] = Dtype(0);
    if (max_positive_1_ >= 0 && max_positive_2_ >= 0) {
      loss[0] += positive_weight_ * (bottom_data[max_positive_1_ * num + max_positive_2_] - positive_margin_);
    }
    if (min_negative_1_ >= 0 && min_negative_2_ >= 0) {
      loss[0] += negative_weight_ * (negative_margin_ - bottom_data[min_negative_1_ * num + min_negative_2_]);
    }
  }
  else {
    loss[0] /= num * (num - 1) / 2;
  }
  if (top.size() >= 2) {
    Dtype* distances = top[1]->mutable_cpu_data();
    distances[0] = positive_distance / positive_count;
    distances[1] = negative_distance / negative_count;
  }
}

template <typename Dtype>
void BatchContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

    caffe_set(num*num, Dtype(0), bottom_diff);
    if (max_only_) {
      if (max_positive_1_ >= 0 && max_positive_2_ >= 0) {
        bottom_diff[max_positive_1_ * num + max_positive_2_] = positive_weight_;
      }
      if (min_negative_1_ >= 0 && min_negative_2_ >= 0) {
        bottom_diff[min_negative_1_ * num + min_negative_2_] = -negative_weight_;
      }
    }
    else {
      for (int i = 0; i < num; ++i) {
        for (int j = i + 1; j < num; ++j) {
          if (label[i] == label[j]) {
            if (bottom_data[i*num + j] > positive_margin_) {
              bottom_diff[i*num + j] = positive_weight_;
            }
          }
          else {
            if (bottom_data[i*num + j] < negative_margin_) {
              bottom_diff[i*num + j] = -negative_weight_;
            }
          }
        }
      }
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (max_only_) {
      caffe_scal(bottom[0]->count(), loss_weight / 2, bottom_diff);
    }
    else {
      caffe_scal(bottom[0]->count(), loss_weight / num, bottom_diff);
    }
  }
}

INSTANTIATE_CLASS(BatchContrastiveLossLayer);
REGISTER_LAYER_CLASS(BatchContrastiveLoss);

}  // namespace caffe
