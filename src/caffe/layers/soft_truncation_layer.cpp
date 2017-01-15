#include <algorithm>
#include <vector>

#include "caffe/layers/soft_truncation_layer.hpp"

namespace caffe {

template <typename Dtype>
void SoftTruncationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype c = this->layer_param_.soft_truncation_param().c();
  for (int i = 0; i < count; ++i) {
    top_data[i] = 1 - exp(bottom_data[i] / (-c));
  }
}

template <typename Dtype>
void SoftTruncationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype c = this->layer_param_.soft_truncation_param().c();
    
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (1 - top_data[i]) / c;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftTruncationLayer);
#endif

INSTANTIATE_CLASS(SoftTruncationLayer);
REGISTER_LAYER_CLASS(SoftTruncation);

}  // namespace caffe
