#include <vector>

#include "caffe/layers/slope_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SlopeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  concat_blob_.Reshape({ 2, height, width });
  Dtype* concat_data = concat_blob_.mutable_cpu_data();
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      concat_data[h * width + w] = (Dtype)h / (Dtype)height;
      concat_data[height * width + h * width + w] = (Dtype)w / (Dtype)width;
    }
  }
}

template <typename Dtype>
void SlopeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape({ bottom[0]->num(), bottom[0]->channels() + 2, bottom[0]->height(), bottom[0]->width() });
}

template <typename Dtype>
void SlopeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* concat_data = concat_blob_.cpu_data();
  int num = bottom[0]->num();
  int channel = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int spatial_dim = height * width;

  for (int n = 0; n < num; n++) {
    caffe_copy(channel * spatial_dim, bottom_data + channel * spatial_dim * n, top_data + (channel + 2) * spatial_dim * n);
    caffe_copy(2 * spatial_dim, concat_data, top_data + (channel + 2) * spatial_dim * n + channel * spatial_dim);
  }
}

template <typename Dtype>
void SlopeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int num = bottom[0]->num();
  int channel = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int spatial_dim = height * width;

  for (int n = 0; n < num; n++) {
    caffe_copy(channel * spatial_dim, top_diff + (channel + 2) * spatial_dim * n, bottom_diff + channel * spatial_dim * n);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SlopeLayer);
#endif

INSTANTIATE_CLASS(SlopeLayer);
REGISTER_LAYER_CLASS(Slope);

}  // namespace caffe
