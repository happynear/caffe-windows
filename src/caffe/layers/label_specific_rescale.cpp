#include <algorithm>
#include <vector>

#include "caffe/layers/label_specific_rescale_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void LabelSpecificRescaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const LabelSpecificRescaleParameter& param = this->layer_param_.label_specific_rescale_param();
    positive_weight = param.positive_weight();
    negative_weight = param.negative_weight();
    positive_lower_bound = param.positive_lower_bound();//Not implemented
    negative_upper_bound = param.negative_upper_bound();//Not implemented
    rescale_test = param.rescale_test();
  }

  template <typename Dtype>
  void LabelSpecificRescaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    if (top[0] != bottom[0]) top[0]->ReshapeLike(*bottom[0]);
  }

template <typename Dtype>
void LabelSpecificRescaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);

  if (!rescale_test && this->phase_ == TEST) return;
  if (positive_weight != Dtype(1.0)) {
    for (int i = 0; i < num; ++i) {
      top_data[i * dim + static_cast<int>(label_data[i])] *= positive_weight;
    }
  }
  if (negative_weight != Dtype(1.0)) {
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; j++) {
        if (j != static_cast<int>(label_data[i])) {
          top_data[i * dim + j] *= negative_weight;
        }
      }
    }
  }
}

template <typename Dtype>
void LabelSpecificRescaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                    const vector<bool>& propagate_down,
                                                    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label_data = bottom[1]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    if (top[0] != bottom[0]) caffe_copy(count, top_diff, bottom_diff);
    if (!rescale_test && this->phase_ == TEST) return;

    if (positive_weight != Dtype(1.0)) {
      for (int i = 0; i < num; ++i) {
        bottom_diff[i * dim + static_cast<int>(label_data[i])] *= positive_weight;
      }
    }
    if (negative_weight != Dtype(1.0)) {
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < dim; j++) {
          if (j != static_cast<int>(label_data[i])) {
            bottom_diff[i * dim + j] *= negative_weight;
          }
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(LabelSpecificRescaleLayer);
#endif

INSTANTIATE_CLASS(LabelSpecificRescaleLayer);
REGISTER_LAYER_CLASS(LabelSpecificRescale);

}  // namespace caffe
