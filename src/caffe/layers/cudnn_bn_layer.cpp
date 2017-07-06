#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_bn_layer.hpp"

#if CUDNN_VERSION_MIN(4, 0, 0)

namespace caffe {

  template <typename Dtype>
  void CuDNNBNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    BNLayer<Dtype>::LayerSetUp(bottom, top);
    if (this->bn_eps_ < CUDNN_BN_MIN_EPSILON) {
      LOG(WARNING) << "bn_eps is set to CUDNN_BN_MIN_EPSILON.";
      // Merely setting as CUDNN_BN_MIN_EPSILON fails the check due to
      // float / double precision problem.
      this->bn_eps_ = CUDNN_BN_MIN_EPSILON * 1.001;
    }
    scale_buf_.ReshapeLike(*(this->blobs_[0]));
    bias_buf_.ReshapeLike(*(this->blobs_[1]));
    save_mean_.ReshapeLike(*(this->blobs_[2]));
    save_inv_variance_.ReshapeLike(*(this->blobs_[3]));

    // Initialize CUDNN.
    CUDNN_CHECK(cudnnCreate(&handle_));
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
    cudnn::createTensor4dDesc<Dtype>(&top_desc_);
    cudnn::createTensor4dDesc<Dtype>(&bn_param_desc_);
    handles_setup_ = true;
  }

  template <typename Dtype>
  void CuDNNBNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
    // Do not call BNLayer::Reshape function as some members are unnecessary
    this->num_ = bottom[0]->num();
    this->channels_ = bottom[0]->channels();
    this->height_ = bottom[0]->height();
    this->width_ = bottom[0]->width();

    top[0]->ReshapeLike(*(bottom[0]));

    // CUDNN tensors
    cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, this->num_, this->channels_,
                                  this->height_, this->width_);
    cudnn::setTensor4dDesc<Dtype>(&top_desc_, this->num_, this->channels_,
                                  this->height_, this->width_);
    // Fix to the spatial mode
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_param_desc_,
                                              bottom_desc_, CUDNN_BATCHNORM_SPATIAL));
  }

  template <typename Dtype>
  CuDNNBNLayer<Dtype>::~CuDNNBNLayer() {
    // Check that handles have been setup before destroying.
    if (!handles_setup_) { return; }

    cudnnDestroyTensorDescriptor(bottom_desc_);
    cudnnDestroyTensorDescriptor(top_desc_);
    cudnnDestroyTensorDescriptor(bn_param_desc_);
    cudnnDestroy(handle_);
  }

  INSTANTIATE_CLASS(CuDNNBNLayer);

}  // namespace caffe
#endif
#endif
