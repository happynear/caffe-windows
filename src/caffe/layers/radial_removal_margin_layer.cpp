#include <algorithm>
#include <vector>

#include "caffe/layers/radial_removal_margin_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void RadialRemovalMarginLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const RadialRemovalMarginParameter& param = this->layer_param_.radial_removal_margin_param();
    scale_ = param.scale();
    original_norm_ = (bottom.size() == 3);
  }

  template <typename Dtype>
  void RadialRemovalMarginLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
    if (top.size() >= 2) {
      top[1]->Reshape({ 4 });
    }
    target_logits_.Reshape({ bottom[0]->num() });
    margins_.Reshape({ bottom[0]->num() });
    max_negative_logits_.Reshape({ bottom[0]->num() });
    sum_exp_negative_logits_.Reshape({ bottom[0]->num() });
    log_sum_exp_negative_logits_.Reshape({ bottom[0]->num() });
    sum_multiplier_.Reshape({ bottom[0]->num() });
    caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
    sum_multiplier_channel_.Reshape({ bottom[0]->channels() });
    caffe_set<Dtype>(sum_multiplier_channel_.count(), Dtype(1), sum_multiplier_channel_.mutable_cpu_data());
    if (!original_norm_) {
      fake_feature_norm_.Reshape({ bottom[0]->num() });
      caffe_set<Dtype>(fake_feature_norm_.count(), scale_, fake_feature_norm_.mutable_cpu_data());
    }
  }

template <typename Dtype>
void RadialRemovalMarginLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void RadialRemovalMarginLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(RadialRemovalMarginLayer);
#endif

INSTANTIATE_CLASS(RadialRemovalMarginLayer);
REGISTER_LAYER_CLASS(RadialRemovalMargin);

}  // namespace caffe
