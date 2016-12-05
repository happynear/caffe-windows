#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  normalize_type_ =
    this->layer_param_.softmax_param().normalize_type();
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  squared_.Reshape(bottom[0]->num(), bottom[0]->channels(),
    bottom[0]->height(), bottom[0]->width());
  norm_.Reshape(bottom[0]->num(), 1,
                  bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* square_data = squared_.mutable_cpu_data();
  int n = bottom[0]->num();
  int d = bottom[0]->count() / n;
  if (normalize_type_ == "L2") {
    caffe_sqr<Dtype>(n*d, bottom_data, square_data);
    for (int i = 0; i<n; ++i) {
      Dtype normsqr = caffe_cpu_asum<Dtype>(d, square_data + i*d) + 1e-6;
      caffe_cpu_scale<Dtype>(d, pow(normsqr, -0.5), bottom_data + i*d, top_data + i*d);
    }
  }
  else if (normalize_type_ == "L1") {
    caffe_abs<Dtype>(n*d, bottom_data, square_data);
    for (int i = 0; i<n; ++i) {
      Dtype normsqr = caffe_cpu_asum<Dtype>(d, square_data + i*d) + 1e-6;
      caffe_cpu_scale<Dtype>(d, 1 / normsqr, bottom_data + i*d, top_data + i*d);
    }
  }
  else {
    NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* square_data = squared_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int n = top[0]->num();
  int d = top[0]->count() / n;
  if (normalize_type_ == "L2") {
    for (int i = 0; i < n; ++i) {
      Dtype a = caffe_cpu_dot(d, top_data + i*d, top_diff + i*d);
      caffe_cpu_scale(d, a, top_data + i*d, bottom_diff + i*d);
      caffe_sub(d, top_diff + i*d, bottom_diff + i*d, bottom_diff + i*d);
      a = caffe_cpu_asum<Dtype>(d, square_data + i*d) + 1e-6;
      caffe_cpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff + i*d, bottom_diff + i*d);
    }
  }
  else if(normalize_type_ == "L1") {
    for (int i = 0; i < n; ++i) {
      Dtype a = caffe_cpu_dot(d, top_data + i*d, top_diff + i*d);
      caffe_cpu_scale(d, a, square_data + i*d, bottom_diff + i*d);
      caffe_sub(d, top_diff + i*d, bottom_diff + i*d, bottom_diff + i*d);
      a = caffe_cpu_asum<Dtype>(d, square_data + i*d) + 1e-6;
      caffe_cpu_scale(d, 1 / a, bottom_diff + i*d, bottom_diff + i*d);
    }
  }
  else {
    NOT_IMPLEMENTED;
  }
}


#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe