#ifndef CAFFE_FOCAL_LOSS_LAYER_HPP_
#define CAFFE_FOCAL_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

  /**
  * Please refer "Focal Loss for Dense Object Detection"
  */
  template <typename Dtype>
  class FocalLossLayer : public LossLayer<Dtype> {
  public:
    explicit FocalLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "FocalLoss"; }
    virtual inline int ExactNumTopBlobs() const { return -1; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 2; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

    void compute_intermediate_values_of_cpu();
    void compute_intermediate_values_of_gpu();

    /// The internal SoftmaxLayer used to map predictions to a distribution.
    shared_ptr<Layer<Dtype> > softmax_layer_;
    /// prob stores the output probability predictions from the SoftmaxLayer.
    Blob<Dtype> prob_;        // softmax output p_t
    Blob<Dtype> log_prob_;    // log(p_t)
    Blob<Dtype> power_prob_;  // alpha * (1 - p_t) ^ gamma
    Blob<Dtype> ones_;        // 1
                              /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
    vector<Blob<Dtype>*> softmax_bottom_vec_;
    /// top vector holder used in call to the underlying SoftmaxLayer::Forward
    vector<Blob<Dtype>*> softmax_top_vec_;
    /// Whether to ignore instances with a certain label.
    bool has_ignore_label_;
    /// The label indicating that an instance should be ignored.
    int ignore_label_;
    /// How to normalize the output loss.
    LossParameter_NormalizationMode normalization_;
    ///
    FocalLossParameter_Type type_;

    Dtype alpha_, beta_, gamma_;
    int softmax_axis_, outer_num_, inner_num_;
    bool one_add_p_;
  };

}  // namespace caffe

#endif  // CAFFE_FOCAL_LOSS_LAYER_HPP_
