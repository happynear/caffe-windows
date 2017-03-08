#ifndef CAFFE_PROPOSAL_LAYERS_HPP_
#define CAFFE_PROPOSAL_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
  template <typename Dtype>
  class ProposalLayer : public Layer<Dtype> {
  public:
    explicit ProposalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top) {
      //LOG(FATAL) << "Reshaping happens during the call to forward.";
    }

    virtual inline const char* type() const { return "ProposalLayer"; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int base_size_;
    int feat_stride_;
    int pre_nms_topn_;
    int post_nms_topn_;
    Dtype nms_thresh_;
    int min_size_;
    Blob<Dtype> anchors_;
    Blob<Dtype> proposals_;
    Blob<int> roi_indices_;
    Blob<int> nms_mask_;
  };

}  // namespace caffe

#endif  // CAFFE_PROPOSAL_LAYERS_HPP_
