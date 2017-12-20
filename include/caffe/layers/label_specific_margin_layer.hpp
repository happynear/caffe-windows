#ifndef CAFFE_LABEL_SPECIFIC_MARGIN_LAYER_HPP_
#define CAFFE_LABEL_SPECIFIC_MARGIN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace caffe {

template <typename Dtype>
class LabelSpecificMarginLayer : public Layer<Dtype> {
 public:
  explicit LabelSpecificMarginLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LabelSpecificMargin"; }
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

  bool has_margin_base_;
  Dtype margin_base_;
  bool has_margin_max_;
  Dtype margin_max_;
  Dtype power_;
  Dtype gamma_;
  int iter_;
  bool margin_on_test_;
  bool auto_tune_;
  bool pass_bp_;
  Blob<Dtype> positive_mask, negative_mask;
  Blob<Dtype> bottom_angle, bottom_square;
  LabelSpecificMarginParameter_MarginType type_;
  Blob<Dtype> theta;
  Blob<Dtype> positive_data;
  Blob<Dtype> sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_LABEL_SPECIFIC_MARGIN_LAYER_HPP_
