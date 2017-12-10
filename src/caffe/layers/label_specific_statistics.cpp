#include <algorithm>
#include <vector>

#include "caffe/layers/label_specific_statistics_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void LabelSpecificStatisticsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    
  }

  template <typename Dtype>
  void LabelSpecificStatisticsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    positive_mask.ReshapeLike(*bottom[0]);
    negative_mask.ReshapeLike(*bottom[0]);
    bottom_square.ReshapeLike(*bottom[0]);
    top[0]->Reshape({ 4 });
    if (top.size() == 2)top[1]->Reshape({ 1 });
  }

template <typename Dtype>
void LabelSpecificStatisticsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void LabelSpecificStatisticsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                    const vector<bool>& propagate_down,
                                                    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(LabelSpecificStatisticsLayer);
#endif

INSTANTIATE_CLASS(LabelSpecificStatisticsLayer);
REGISTER_LAYER_CLASS(LabelSpecificStatistics);

}  // namespace caffe
