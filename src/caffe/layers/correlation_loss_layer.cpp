#include <algorithm>
#include <vector>

#include "caffe/layers/correlation_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void CorrelationLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_GE(bottom.size(), 2);
}

template <typename Dtype>
void CorrelationLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
}

template <typename Dtype>
void CorrelationLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = caffe_cpu_dot(bottom[0]->count(), bottom_data, label);
  loss[0] /= bottom[0]->count();
}

template <typename Dtype>
void CorrelationLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    caffe_cpu_scale(bottom[0]->count(), top[0]->cpu_diff()[0] / bottom[0]->count(), bottom[1]->cpu_data(), bottom[0]->mutable_cpu_data());
  }
  if (propagate_down[1]) {
    caffe_cpu_scale(bottom[1]->count(), top[0]->cpu_diff()[0] / bottom[0]->count(), bottom[0]->cpu_data(), bottom[1]->mutable_cpu_data());
  }
}

#ifdef CPU_ONLY
STUB_GPU(CorrelationLossLayer);
#endif

INSTANTIATE_CLASS(CorrelationLossLayer);
REGISTER_LAYER_CLASS(CorrelationLoss);

}  // namespace caffe
