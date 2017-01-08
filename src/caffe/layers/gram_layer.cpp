#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {

  template <typename Dtype>
  void GramLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->channels(), 1);
  }

  template <typename Dtype>
  void GramLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int num = bottom[0]->num();
    int channel = bottom[0]->channels();
    int spatial_dim = bottom[0]->height() * bottom[0]->width();
    Blob<Dtype> temp;
    temp.ReshapeLike(*bottom[0]);
    caffe_copy<Dtype>(bottom[0]->count(), bottom_data, temp.mutable_cpu_data());

    for (int n = 0; n < num; n++) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channel, channel, spatial_dim,
        1 / (Dtype)spatial_dim / (Dtype)channel, bottom_data + n * spatial_dim * channel, temp.cpu_data() + n * spatial_dim * channel, Dtype(0), top_data + n * channel * channel);
    }
  }

  template <typename Dtype>
  void GramLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    Dtype* top_diff = top[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->shape(0);
    int channel = bottom[0]->shape(1);
    int spatial_dim = bottom[0]->shape(2) * bottom[0]->shape(3);

    for (int n = 0; n < num; n++) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channel, spatial_dim, channel,
                            1 / (Dtype)spatial_dim / (Dtype)channel, top_diff + n * channel * channel, bottom_data + n * spatial_dim * channel,
                            Dtype(0), bottom_diff + n * spatial_dim * channel);
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channel, spatial_dim, channel,
                            1 / (Dtype)spatial_dim / (Dtype)channel, top_diff + n * channel * channel, bottom_data + n * spatial_dim * channel,
                            Dtype(1), bottom_diff + n * spatial_dim * channel);
    }
  }


#ifdef CPU_ONLY
  STUB_GPU(GramLayer);
#endif

  INSTANTIATE_CLASS(GramLayer);
  REGISTER_LAYER_CLASS(Gram);

}  // namespace caffe
