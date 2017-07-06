#include <algorithm>
#include <vector>
#include <math.h>
#include <float.h>

#include "caffe/layers/general_triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void GeneralTripletLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_GE(bottom.size(), 2);
  margin_ = this->layer_param_.general_triplet_loss_param().margin();
  add_center_loss_ = this->layer_param_.general_triplet_loss_param().add_center_loss();
  hardest_only_ = this->layer_param_.general_triplet_loss_param().hardest_only();
  positive_weight_ = this->layer_param_.general_triplet_loss_param().positive_weight();
  negative_weight_ = this->layer_param_.general_triplet_loss_param().negative_weight();
  positive_first_ = this->layer_param_.general_triplet_loss_param().positive_first();
  positive_upper_bound_ = this->layer_param_.general_triplet_loss_param().positive_upper_bound();
}

template <typename Dtype>
void GeneralTripletLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  if (top.size() >= 2) {
    // positive distance, negative distance.
    top[1]->Reshape({ 2 });
  }
  if (hardest_only_) {
    hardest_index_.Reshape({ bottom[0]->num() });
  }
}

template <typename Dtype>
void GeneralTripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int* hardest_index_data = NULL;
  if (hardest_only_) hardest_index_data = hardest_index_.mutable_cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  Dtype positive_distance = Dtype(0);
  Dtype negative_distance = Dtype(0);

  for (int i = 0; i < num; ++i) {
    Dtype same_distance = bottom_data[i * dim + static_cast<int>(label[i])];
    positive_distance += same_distance;
    if (hardest_only_) hardest_index_data[i] = -1;
    for (int j = 0; j < dim; ++j) {
      if (j != static_cast<int>(label[i])) {
        bottom_diff[i * dim + j] = std::max(
          Dtype(0), same_distance + margin_ - bottom_data[i * dim + j]) * negative_weight_;
        negative_distance += bottom_data[i * dim + j];
        if (hardest_only_ && bottom_diff[i * dim + j] > 0 &&
          (hardest_index_data[i] < 0 || bottom_diff[i * dim + j] > bottom_diff[i * dim + hardest_index_data[i]])) {
          hardest_index_data[i] = j;
        }
      }
      else {
        bottom_diff[i * dim + j] = Dtype(0);
      }
    }
    if (hardest_only_ && hardest_index_data[i] >= 0) {
      for (int j = 0; j < dim; ++j) {
        if (j != static_cast<int>(label[i]) && j != hardest_index_data[i]) {
          bottom_diff[i * dim + j] = Dtype(0);
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
  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
  if (top.size() >= 2) {
    Dtype* distances = top[1]->mutable_cpu_data();
    distances[0] = positive_distance / num;
    distances[1] = negative_distance / num / (dim - 1);
  }
}

template <typename Dtype>
void GeneralTripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int* hardest_index_data = NULL;
    if (hardest_only_) hardest_index_data = hardest_index_.mutable_cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;  

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
      int negative_sum = 0;
      if (hardest_only_) {
        if (hardest_index_data[i] >= 0 && bottom_data[i * dim + hardest_index_data[i]]) {
          bottom_diff[i * dim + hardest_index_data[i]] = -negative_weight_;
          negative_sum += 1;
        }
      }
      else {
        for (int j = 0; j < dim; ++j) {
          if (j != static_cast<int>(label[i])) {
            if (same_distance + margin_ > bottom_data[i * dim + j]) {
              bottom_diff[i * dim + j] = -negative_weight_;
              negative_sum += 1;
            }
            else {
              bottom_diff[i * dim + j] = Dtype(0);
            }
          }
        }
      }
      
      if (add_center_loss_ || negative_sum > 0) {
        bottom_diff[i * dim + static_cast<int>(label[i])] = positive_weight_;
      }
    } 

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    //Dtype weighted_count = num * (abs(positive_weight_) + (dim - 1)*abs(negative_weight_));
    if (bottom.size() == 3) {
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < dim; ++j) {
          bottom_diff[i * dim + j] *= bottom[2]->cpu_data()[i];
        }
      }
    }
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(GeneralTripletLossLayer);
REGISTER_LAYER_CLASS(GeneralTripletLoss);

}  // namespace caffe
