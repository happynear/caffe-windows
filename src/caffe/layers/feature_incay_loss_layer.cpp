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
  CHECK_EQ(bottom.size(), 3);
}

template <typename Dtype>
void FeatureIncayLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void FeatureIncayLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void FeatureIncayLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    
  }
}

INSTANTIATE_CLASS(FeatureIncayLossLayer);
REGISTER_LAYER_CLASS(FeatureIncayLoss);

}  // namespace caffe
