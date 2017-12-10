#ifndef CAFFE_CHANNEL_SCALE_LAYER_HPP_
#define CAFFE_CHANNEL_SCALE_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
  /**
  * @brief ChannelScales input.
  */
  template <typename Dtype>
  class ChannelScaleLayer : public Layer<Dtype> {
  public:
    explicit ChannelScaleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "ChannelScale"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    Blob<Dtype> sum_multiplier_;
    bool do_forward_;
    bool do_backward_feature_;
    bool do_backward_scale_;
    bool global_scale_;
    Dtype max_global_scale_;
    Dtype min_global_scale_;
  };

}

#endif  // CAFFE_CHANNEL_SCALE_LAYER_HPP_
