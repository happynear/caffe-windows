#include <algorithm>
#include <vector>
#include <limits>

#include "caffe/filler.hpp"
#include "caffe/layers/global_lse_margin_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void GlobalLSEMarginLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
    original_norm_ = (bottom.size() == 3);
    const ScaleParameter& param = this->layer_param_.scale_param();
    this->blobs_.resize(2);
    this->blobs_[0].reset(new Blob<Dtype>({ 1 }));// s
    FillerParameter filler_param(param.filler());
    if (!param.has_filler()) {
      this->blobs_[0]->mutable_cpu_data()[0] = Dtype(30);
    }
    else {
      shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
      filler->Fill(this->blobs_[0].get());
    }
    if (param.has_min_value())min_scale_ = param.min_value();
    else min_scale_ = 0;
    if(param.has_max_value()) max_scale_ = param.max_value();
    else max_scale_ = std::numeric_limits<float>::max();

    this->blobs_[1].reset(new Blob<Dtype>({ 1 }));// m
    this->blobs_[1]->mutable_cpu_data()[0] = Dtype(0);

    if (this->layer_param_.param_size() == 0) {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      if (original_norm_) {
        fixed_param_spec->set_lr_mult(0.f);
      }
      else {
        fixed_param_spec->set_lr_mult(1.f);
      }
      fixed_param_spec->set_decay_mult(0.f);
    }
    if (this->layer_param_.param_size() == 1) {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      fixed_param_spec->set_lr_mult(0.f);
      fixed_param_spec->set_decay_mult(0.f);
    }
  }

  template <typename Dtype>
  void GlobalLSEMarginLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
    if (top.size() >= 2) {
      top[1]->Reshape({ 5 });
    }
    target_logits_.Reshape({ bottom[0]->num() });
    target_logits_square_.Reshape({ bottom[0]->num() });
    margins_.Reshape({ bottom[0]->num() });
    lse_.Reshape({ bottom[0]->num() });
    max_negative_logits_.Reshape({ bottom[0]->num() });
    sum_multiplier_.Reshape({ bottom[0]->num() });
    caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
    sum_multiplier_channel_.Reshape({ bottom[0]->channels() });
    caffe_set<Dtype>(sum_multiplier_channel_.count(), Dtype(1.0), sum_multiplier_channel_.mutable_cpu_data());
  }

template <typename Dtype>
void GlobalLSEMarginLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void GlobalLSEMarginLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
template <typename Dtype>
Dtype GlobalLSEMarginLayer<Dtype>::MeanShift(Blob<Dtype>* margins, Dtype mu, int loop) {
  NOT_IMPLEMENTED;
  return Dtype(0);
}

template double GlobalLSEMarginLayer<double>::MeanShift(Blob<double>* margins, double mu, int loop);
template float GlobalLSEMarginLayer<float>::MeanShift(Blob<float>* margins, float mu, int loop);

template <typename Dtype>
Dtype GlobalLSEMarginLayer<Dtype>::MeanMaxNegativeLogit(const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
  return Dtype(0);
}

template double GlobalLSEMarginLayer<double>::MeanMaxNegativeLogit(const vector<Blob<double>*>& bottom);
template float GlobalLSEMarginLayer<float>::MeanMaxNegativeLogit(const vector<Blob<float>*>& bottom);

template <typename Dtype>
Dtype GlobalLSEMarginLayer<Dtype>::CalcLSE(const vector<Blob<Dtype>*>& bottom, Blob<Dtype>* LSE) {
  NOT_IMPLEMENTED;
  return Dtype(0);
}

template double GlobalLSEMarginLayer<double>::CalcLSE(const vector<Blob<double>*>& bottom, Blob<double>* LSE);
template float GlobalLSEMarginLayer<float>::CalcLSE(const vector<Blob<float>*>& bottom, Blob<float>* LSE);
#endif


#ifdef CPU_ONLY
STUB_GPU(GlobalLSEMarginLayer);
#endif

INSTANTIATE_CLASS(GlobalLSEMarginLayer);
REGISTER_LAYER_CLASS(GlobalLSEMargin);

}  // namespace caffe
