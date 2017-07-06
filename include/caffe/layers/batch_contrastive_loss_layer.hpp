#ifndef CAFFE_BATCH_CONTRASTIVE_LOSS_LAYER_HPP_
#define CAFFE_BATCH_CONTRASTIVE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class BatchContrastiveLossLayer : public LossLayer<Dtype> {
 public:
  explicit BatchContrastiveLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "BatchContrastiveLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype positive_margin_;
  Dtype negative_margin_;
  Dtype positive_weight_;
  Dtype negative_weight_;
  bool max_only_;
  int max_positive_1_, max_positive_2_;
  int min_negative_1_, min_negative_2_;
};


}  // namespace caffe

#endif  // CAFFE_BATCH_CONTRASTIVE_LOSS_LAYER_HPP_
