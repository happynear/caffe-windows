#include <functional>
#include <utility>
#include <vector>
#include <limits>

#include "caffe/layers/fcn_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void FCNAccuracyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    top_k_ = this->layer_param_.accuracy_param().top_k();
  }

  template <typename Dtype>
  void FCNAccuracyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->count(), bottom[1]->count());
    if (bottom.size() == 3) {
      CHECK_EQ(bottom[1]->count(), bottom[2]->count())
        << "Number of loss weights must match number of label.";
    }
    vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
    top[0]->Reshape(top_shape);
    if (top.size() > 1) {
      // Per-class accuracy is a vector; 1 axes.
      vector<int> top_shape_per_class(1);
      top_shape_per_class[0] = bottom[0]->shape(label_axis_);
      top[1]->Reshape(top_shape_per_class);
      nums_buffer_.Reshape(top_shape_per_class);
    }
  }

  template <typename Dtype>
  void FCNAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Dtype accuracy = 0;
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    int count = 0;
    Dtype max_val;
    int max_n = 0, max_c = 0, max_h = 0, max_w = 0;
    for (int n = 0; n < bottom[0]->num(); n++) {
      if (n % top_k_ == 0) {
        max_val = -std::numeric_limits<Dtype>::max();
        max_h = 0;
        max_w = 0;
        max_n = 0;
      }
      for (int c = 0; c < bottom[0]->channels(); c++) {
        for (int h = 0; h < bottom[0]->height(); h++) {
          for (int w = 0; w < bottom[0]->width(); w++) {
            if (bottom_data[bottom[0]->offset(n, c, h, w)] > max_val) {
              max_val = bottom_data[bottom[0]->offset(n, c, h, w)];
              max_h = h;
              max_w = w;
              max_c = c;
              max_n = n % top_k_;
            }
          }
        }
      }
      if (n % top_k_ == top_k_ - 1) {
        if (max_n == (top_k_ - 1) / 2 && bottom_label[bottom[1]->offset(max_n, max_c, max_h, max_w)] > 0) accuracy += (Dtype)1.0;
        count++;
      }
    }

    // LOG(INFO) << "Accuracy: " << accuracy;
    top[0]->mutable_cpu_data()[0] = accuracy / count;
    // Accuracy layer should not be used as a loss function.
  }

  INSTANTIATE_CLASS(FCNAccuracyLayer);
  REGISTER_LAYER_CLASS(FCNAccuracy);

}  // namespace caffe
