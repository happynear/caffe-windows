#ifndef CAFFE_XXX_LAYER_HPP_
#define CAFFE_XXX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

  /**
  * @brief Produce point heat maps by a series of the input 2d coordinates.
  *
  * TODO(dox): thorough documentation for Forward, Backward, and proto params.
  */
  template <typename Dtype>
  class HotspotLayer : public Layer<Dtype> {
  public:
    explicit HotspotLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Hotspot"; }
    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int ExactTopBlobs() const { return 1; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int height_;
    int width_;
    Dtype gaussian_std_;
    int data_height_;
    int data_width_;
    bool mean_removed_;
  };

}  // namespace caffe

#endif  // CAFFE_XXX_LAYER_HPP_
