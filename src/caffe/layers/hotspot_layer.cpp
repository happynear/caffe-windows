#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/hotspot_layer.hpp"
#define CV_PI 3.1415926535897932384626433832795
#define GAUSSIAN(x0,y0,x,y) 0.5 / gaussian_std_ / gaussian_std_ / CV_PI * exp(-0.5 * (((x0)-(x)) * ((x0)-(x)) + ((y0)-(y)) * ((y0)-(y))) / gaussian_std_ / gaussian_std_)

namespace caffe {

  const float kEps = 1e-4;

template <typename Dtype>
void HotspotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  HotspotParameter hotspot_param = this->layer_param_.hotspot_param();
  height_ = hotspot_param.output_height();
  width_ = hotspot_param.output_width();
  gaussian_std_ = hotspot_param.gaussian_std();
  data_height_ = hotspot_param.data_height();
  data_width_ = hotspot_param.data_width();
  mean_removed_ = hotspot_param.mean_removed();
}

template <typename Dtype>
void HotspotLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels() % 2, 0) << "The coordinate blob's size must be able to divided by 2!";
  CHECK(data_width_ * data_height_ > 0) << "The data layers' size must be specified.";
  int num_point = bottom[0]->channels() / 2;
  if (height_ * width_ == 0) {
    CHECK(bottom.size() == 2) << "When the ooutput_height and output_width is not mannually set, "
      << "one more bottom blob is needed to inference the size of the output.";
    height_ = bottom[1]->height();
    width_ = bottom[1]->width();
  }
  top[0]->Reshape({ bottom[0]->num(), num_point, height_, width_ });
}

template <typename Dtype>
void HotspotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* point_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  const int num_point = bottom[0]->shape(1) / 2;
  const int num = bottom[0]->num();
  Dtype temp;
  for (int n = 0; n < num; n++) {
    for (int i = 0; i < num_point; i++) {
      float p1 = (point_data[n * num_point * 2 + 2 * i] / (Dtype)data_width_ + (mean_removed_ ? 0.5 : 0)) * (Dtype)width_;
      float p2 = (point_data[n * num_point * 2 + 2 * i + 1] / (Dtype)data_height_ + (mean_removed_ ? 0.5 : 0)) * (Dtype)height_;
      LOG(INFO)<< "n:"<<n<<" p:"<<i<< " p1:" << p1 << " p2:" << p2;
      for (int h = 0; h < height_; h++) {
        for (int w = 0; w < width_; w++) {
          temp = GAUSSIAN(p1, p2, (Dtype)w, (Dtype)h);
          if (temp > kEps) {
            top_data[top[0]->offset(n, i, h, w)] = temp;
          }
          else {
            top_data[top[0]->offset(n, i, h, w)] = 0;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void HotspotLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // Acatually, we can get gradient from the top_diff for the coordinate values.
  // However, I can't imagine in which scene this could be useful.
  // I will implement it one day I come up with some ideas to utilize the gradients.
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(HotspotLayer);
#endif

INSTANTIATE_CLASS(HotspotLayer);
REGISTER_LAYER_CLASS(Hotspot);

}  // namespace caffe
