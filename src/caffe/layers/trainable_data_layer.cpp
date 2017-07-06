#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/trainable_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TrainableDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else{
    const TrainableDataParameter& param = this->layer_param_.trainable_data_param();
    vector<int> blob_shape(param.shape().begin(), param.shape().end());
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(blob_shape));
    LOG(INFO) << "Trainable data shape: " << this->blobs_[0]->shape_string();
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(param.blob_filler()));
    filler->Fill(this->blobs_[0].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void TrainableDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*this->blobs_[0]);
}

template <typename Dtype>
void TrainableDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*this->blobs_[0]);
}

template <typename Dtype>
void TrainableDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    this->blobs_[0]->ShareDiff(*top[0]);
  }
}

#ifdef CPU_ONLY
STUB_GPU(TrainableDataLayer);
#endif

INSTANTIATE_CLASS(TrainableDataLayer);
REGISTER_LAYER_CLASS(TrainableData);

}  // namespace caffe
