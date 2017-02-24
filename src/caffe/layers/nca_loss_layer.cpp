#include <algorithm>
#include <vector>

#include "caffe/layers/nca_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void NCALossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  min_negative_only_ = this->layer_param_.nca_param().min_negative_only();
  CHECK_GE(bottom.size(), 2);
}

template <typename Dtype>
void NCALossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  if (top.size() == 2) {
    // positive distance, negative distance.
    top[1]->Reshape({ 2 });
  }
  if (min_negative_only_) {
    min_negative_index_.Reshape({ bottom[0]->num() });
  }
}

template <typename Dtype>
void NCALossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int* min_negative_index_data = NULL;
  if (min_negative_only_) min_negative_index_data = min_negative_index_.mutable_cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  Dtype positive_distance = Dtype(0);
  Dtype negative_distance = Dtype(0);
  Dtype* loss = top[0]->mutable_cpu_data();

  for (int i = 0; i < num; ++i) {
    if (min_negative_only_) min_negative_index_data[i] = 0;
    for (int j = 0; j < dim; ++j) {
      if (j == static_cast<int>(label[i])) {
        positive_distance += bottom_data[i * dim + j];
        loss[0] += bottom_data[i * dim + j];
      }
      else {
        negative_distance += bottom_data[i * dim + j];
        if(!min_negative_only_) loss[0] += exp(-bottom_data[i * dim + j]);
        if (min_negative_only_ && bottom_data[i * dim + j] < bottom_data[i*dim + min_negative_index_data[i]]) {
          min_negative_index_data[i] = j;
        }
      }
    }
    if (min_negative_only_) {
      loss[0] += exp(-bottom_data[i * dim + min_negative_index_data[i]]);
    }
  }
  
  loss[0] /= num;
  if (top.size() >= 2) {
    Dtype* distances = top[1]->mutable_cpu_data();
    distances[0] = positive_distance / num;
    distances[1] = negative_distance / num / (dim - 1);
  }
}

template <typename Dtype>
void NCALossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    const int* min_negative_index_data = NULL;
    if (min_negative_only_) min_negative_index_data = min_negative_index_.cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    Dtype negative_sum = Dtype(0);

    for (int i = 0; i < num; ++i) {
      if (min_negative_only_) {
        bottom_diff[i * dim + static_cast<int>(label[i])] = 1;
        bottom_diff[i * dim + min_negative_index_data[i]] = -1 * exp(-bottom_data[i * dim + min_negative_index_data[i]]);
      }
      else {
        for (int j = 0; j < dim; ++j) {
          if (j == static_cast<int>(label[i])) {
            bottom_diff[i * dim + j] = 1;
          }
          else {
            bottom_diff[i * dim + j] = -1 * exp(-bottom_data[i * dim + j]);
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

INSTANTIATE_CLASS(NCALossLayer);
REGISTER_LAYER_CLASS(NCALoss);

}  // namespace caffe
