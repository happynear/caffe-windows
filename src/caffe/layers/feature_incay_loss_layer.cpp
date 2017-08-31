#include <algorithm>
#include <vector>
#include <math.h>
#include <float.h>

#include "caffe/layers/feature_incay_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void FeatureIncayLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom.size(), 4);//feature, norm, score, label
  force_incay_ = this->layer_param_.feature_incay_loss_param().force_incay();
}

template <typename Dtype>
void FeatureIncayLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  well_classified.Reshape({ bottom[0]->num() });
  if (top.size() == 2) top[1]->Reshape({ 1 });
}

template <typename Dtype>
void FeatureIncayLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //const Dtype* feature = bottom[0]->cpu_data();
  const Dtype* norm = bottom[1]->cpu_data();
  const Dtype* score = bottom[2]->cpu_data();
  const Dtype* label = bottom[3]->cpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count(1);
  const int label_num = bottom[2]->count(1);
  Dtype* loss = top[0]->mutable_cpu_data();

  Dtype epsilon = Dtype(1e-6);
  loss[0] = Dtype(0.0);
  if (top.size() == 2) {
    top[1]->mutable_cpu_data()[0] = Dtype(0.0);
  }

  for (int n = 0; n < num; n++) {
    int max_score_pos = 0;
    if (!force_incay_) {
      for (int l = 1; l < label_num; l++) {
        if (score[n*label_num + l] > score[n*label_num + max_score_pos]) {
          max_score_pos = l;
        }
      }
    }
    if (force_incay_ || max_score_pos == static_cast<int>(label[n])) {
      loss[0] += 1 / norm[n] / norm[n]; 
      well_classified.mutable_cpu_data()[n] = 1;
    }
    else {
      well_classified.mutable_cpu_data()[n] = 0;
    }
    if (top.size() == 2) {
      top[1]->mutable_cpu_data()[0] += norm[n];
    }
  }
  loss[0] /= num;
  top[1]->mutable_cpu_data()[0] /= num;
}

template <typename Dtype>
void FeatureIncayLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]|| propagate_down[2]|| propagate_down[3]) {
    LOG(FATAL) << this->type()
               << " Layer can only backpropagate to feature inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* feature = bottom[0]->cpu_data();
    Dtype* feature_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* norm = bottom[1]->cpu_data();
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count(1);
    const Dtype loss_weight = top[0]->cpu_diff()[0];

    for (int n = 0; n < num; n++) {
      if (well_classified.cpu_data()[n]) {
        caffe_cpu_scale<Dtype>(dim, -2 * loss_weight / pow(norm[n], 4), feature + n*dim, feature_diff + n*dim);
      }
      else {
        caffe_set<Dtype>(dim, Dtype(0), feature_diff + n*dim);
      }
    }
  }
}

INSTANTIATE_CLASS(FeatureIncayLossLayer);
REGISTER_LAYER_CLASS(FeatureIncayLoss);

}  // namespace caffe
