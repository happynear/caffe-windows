#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/arccos_layer.hpp"

namespace caffe {

template <typename Dtype>
void ArccosLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  
  for (int i = 0; i < count; ++i) {
    top_data[i] = acosf(bottom_data[i]);
  }
}

template <typename Dtype>
void ArccosLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();

  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      Dtype fixed_in_data = std::min(bottom_data[i], Dtype(1.0) - Dtype(0.01));
      bottom_diff[i] = top_diff[i] * -1 / sqrtf(1.0f - fixed_in_data * fixed_in_data);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ArccosLayer);
#endif

INSTANTIATE_CLASS(ArccosLayer);
REGISTER_LAYER_CLASS(Arccos);

}  // namespace caffe
