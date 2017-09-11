#ifndef CAFFE_LABEL_SPECIFIC_HARD_MARGIN_LAYER_HPP_
#define CAFFE_LABEL_SPECIFIC_HARD_MARGIN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace caffe {

template <typename Dtype>
class LabelSpecificHardMarginLayer : public Layer<Dtype> {
 public:
  explicit LabelSpecificHardMarginLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LabelSpecificHardMargin"; }
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

  Blob<int> hardest_pos_;
  Blob<Dtype> margins_;
  Blob<Dtype> sum_multiplier_;
  Dtype positive_weight;
};

}  // namespace caffe

#endif  // CAFFE_LABEL_SPECIFIC_HARD_MARGIN_LAYER_HPP_
