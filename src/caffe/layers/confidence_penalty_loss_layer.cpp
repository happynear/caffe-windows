#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/confidence_penalty_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void ConfidencePenaltyLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    beta_ = this->layer_param_.confidence_penalty_loss_param().beta();
  }

  template <typename Dtype>
  void ConfidencePenaltyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->height(), 1);
    CHECK_EQ(bottom[1]->width(), 1);
  }

  template <typename Dtype>
  void ConfidencePenaltyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    Dtype loss = 0;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; ++j) {
        Dtype prob = std::max(
          bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
        if (j == static_cast<int>(bottom_label[i])) {
          loss -= (Dtype(1.0) - beta_) * log(prob);
        }
        else {
          loss -= (beta_ / (num - Dtype(1.0))) * log(prob);
        }
      }
    }
    top[0]->mutable_cpu_data()[0] = loss / num;
  }

  template <typename Dtype>
  void ConfidencePenaltyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
      const Dtype* bottom_data = bottom[0]->cpu_data();
      const Dtype* bottom_label = bottom[1]->cpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      int num = bottom[0]->num();
      int dim = bottom[0]->count() / bottom[0]->num();
      caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
      const Dtype scale = -top[0]->cpu_diff()[0] / num;
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < dim; ++j) {
          Dtype prob = std::max(bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
          if (j == static_cast<int>(bottom_label[i])) {
            bottom_diff[i * dim + j] = scale * (Dtype(1.0) - beta_) / prob;
          }
          else {
            bottom_diff[i * dim + j] = scale * (beta_ / (num - Dtype(1.0))) / prob;
          }
        }
      }
    }
  }

  INSTANTIATE_CLASS(ConfidencePenaltyLossLayer);
  REGISTER_LAYER_CLASS(ConfidencePenaltyLoss);

}  // namespace caffe
