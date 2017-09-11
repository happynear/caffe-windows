#include <algorithm>
#include <vector>

#include "caffe/layers/label_specific_hard_margin.hpp"

namespace caffe {

  template <typename Dtype>
  void LabelSpecificHardMarginLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const LabelSpecificHardMarginParameter& param = this->layer_param_.label_specific_hard_margin_param();
    positive_weight = param.positive_weight();
    //transform_test_ = param.transform_test() & (this->phase_ == TRAIN);
  }

  template <typename Dtype>
  void LabelSpecificHardMarginLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
    if (top.size() == 2) {
      top[1]->Reshape({ 1 });
    }
    hardest_pos_.Reshape({ bottom[0]->num() });
    margins_.Reshape({ bottom[0]->num() });
    if (sum_multiplier_.Reshape({ bottom[0]->channels() })) {
      caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
    }
  }

template <typename Dtype>
void LabelSpecificHardMarginLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void LabelSpecificHardMarginLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(LabelSpecificHardMarginLayer);
#endif

INSTANTIATE_CLASS(LabelSpecificHardMarginLayer);
REGISTER_LAYER_CLASS(LabelSpecificHardMargin);

}  // namespace caffe
