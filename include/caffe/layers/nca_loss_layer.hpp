#ifndef CAFFE_NCA_LOSS_LAYER_HPP_
#define CAFFE_NCA_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class NCALossLayer : public LossLayer<Dtype> {
 public:
  explicit NCALossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "NCALoss"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  /// @copydoc NCALossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool min_negative_only_;
  Blob<int> min_negative_index_;
};
}  // namespace caffe

#endif  // CAFFE_NCA_LOSS_LAYER_HPP_
