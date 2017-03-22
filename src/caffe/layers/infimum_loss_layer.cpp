#include <algorithm>
#include <vector>

#include "caffe/layers/infimum_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void InfimumLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_GE(bottom.size(), 1);
  infimum_ = this->layer_param_.infimum_loss_param().infimum();
}

template <typename Dtype>
void InfimumLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void InfimumLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  int count = bottom[0]->count();
  Dtype* loss = top[0]->mutable_cpu_data();

  loss[0] = Dtype(0);
  for (int i = 0; i < count; i++) {
    if(bottom_data[i] < infimum_) {
      loss[0] += infimum_ - bottom_data[i];
    }
  }

  loss[0] /= count;
}

template <typename Dtype>
void InfimumLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int count = bottom[0]->count();
    for (int i = 0; i < count; i++) {
      if (bottom_data[i] < infimum_) {
        bottom_diff[i] = -1 * top[0]->cpu_diff()[0] / count;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InfimumLossLayer);
#endif

INSTANTIATE_CLASS(InfimumLossLayer);
REGISTER_LAYER_CLASS(InfimumLoss);

}  // namespace caffe
