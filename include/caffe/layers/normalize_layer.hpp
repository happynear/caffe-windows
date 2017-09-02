#ifndef CAFFE_NORMALIZE_LAYER_HPP_
#define CAFFE_NORMALIZE_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
  /**
  * @brief Normalizes input.
  */
  template <typename Dtype>
  class NormalizeLayer : public Layer<Dtype> {
  public:
    explicit NormalizeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Normalize"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
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

    Blob<Dtype> sum_multiplier_, squared_, norm_;
    std::string normalize_type_;
    bool fix_gradient_;
    bool bp_norm_;
  };

}

#endif  // CAFFE_NORMALIZE_LAYER_HPP_
