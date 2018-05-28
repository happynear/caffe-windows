#ifndef CAFFE_LABEL_SPECIFIC_AFFINE_LAYER_HPP_
#define CAFFE_LABEL_SPECIFIC_AFFINE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace caffe {

template <typename Dtype>
class GlobalLSEMarginLayer : public Layer<Dtype> {
 public:
  explicit GlobalLSEMarginLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GlobalLSEMargin"; }
  virtual inline int MinNumBottomBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual Dtype CalcLSE(const vector<Blob<Dtype>*>& bottom, Blob<Dtype>* LSE);
  virtual Dtype MeanMaxNegativeLogit(const vector<Blob<Dtype>*>& bottom);
  virtual Dtype MeanShift(Blob<Dtype>* margins, Dtype mu, int loop = 1);

  Dtype scale_;
  Dtype min_scale_;
  Dtype max_scale_;
  bool original_norm_;
  Blob<Dtype> target_logits_;
  Blob<Dtype> target_logits_square_;
  Blob<Dtype> margins_;
  Blob<Dtype> max_negative_logits_;
  Blob<Dtype> lse_;
  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> sum_multiplier_channel_;
  Blob<Dtype> fake_feature_norm_;
};

}  // namespace caffe

#endif  // CAFFE_LABEL_SPECIFIC_AFFINE_LAYER_HPP_
