#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/trainable_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TrainableDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*this->blobs_[0]);
}

template <typename Dtype>
void TrainableDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    this->blobs_[0]->ShareDiff(*top[0]);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TrainableDataLayer);

}  // namespace caffe
