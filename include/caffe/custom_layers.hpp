#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	/**
	* @brief Randomized Leaky Rectified Linear Unit @f$
	*        y_i = \max(0, x_i) + frac{\min(0, x_i)}{a_i}
	*        @f$. The negative slope is randomly generated from
	*        uniform distribution U(lb, ub).
	*/
	template <typename Dtype>
	class InsanityLayer : public NeuronLayer<Dtype> {
	public:
		/**
		* @param param provides InsanityParameter insanity_param,
		*/
		explicit InsanityLayer(const LayerParameter& param)
			: NeuronLayer<Dtype>(param) {}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Insanity"; }

	protected:
		/**
		* @param bottom input Blob vector (length 1)
		*   -# @f$ (N \times C \times ...) @f$
		*      the inputs @f$ x @f$
		* @param top output Blob vector (length 1)
		*   -# @f$ (N \times C \times ...) @f$
		*      the computed outputs for each channel @f$i@f$ @f$
		*        y_i = \max(0, x_i) + a_i \min(0, x_i)
		*      @f$.
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


		Dtype lb_, ub_, mean_slope;
		Blob<Dtype> alpha;  // random generated negative slope
		Blob<Dtype> bottom_memory_;  // memory for in-place computation
	};

	/* ROIPoolingLayer - Region of Interest Pooling Layer
	*/
	template <typename Dtype>
	class ROIPoolingLayer : public Layer<Dtype> {
	public:
		explicit ROIPoolingLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ROIPooling"; }

		virtual inline int MinBottomBlobs() const { return 2; }
		virtual inline int MaxBottomBlobs() const { return 2; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int channels_;
		int height_;
		int width_;
		int pooled_height_;
		int pooled_width_;
		Dtype spatial_scale_;
		Blob<int> max_idx_;
	};

	template <typename Dtype>
	class LocalLayer : public Layer<Dtype> {
	public:
		explicit LocalLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Local"; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline bool EqualNumBottomTopBlobs() const { return true; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


		int kernel_size_;
		int stride_;
		int num_;
		int channels_;
		int pad_;
		int height_, width_;
		int height_out_, width_out_;
		int num_output_;
		bool bias_term_;

		int M_;
		int K_;
		int N_;

		Blob<Dtype> col_buffer_;
	};

	template <typename Dtype>
	class SmoothL1LossLayer : public LossLayer<Dtype> {
	public:
		explicit SmoothL1LossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), diff_() {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "SmoothL1Loss"; }

		virtual inline int ExactNumBottomBlobs() const { return -1; }
		virtual inline int MinBottomBlobs() const { return 2; }
		virtual inline int MaxBottomBlobs() const { return 3; }

		/**
		* Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
		* to both inputs -- override to return true and always allow force_backward.
		*/
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> diff_;
		Blob<Dtype> errors_;
		bool has_weights_;
    Dtype turn_point_;
	};

  /* Yuanyang adding triplet loss layer */
  /* *
  * * @brief Computes the triplet loss
  * */
  template <typename Dtype>
  class TripletLossLayer : public LossLayer<Dtype> {
  public:
    explicit TripletLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
    virtual inline int ExactNumBottomBlobs() const { return 3; }
    virtual inline const char* type() const { return "TripletLoss"; }
    /* *
    * * Unlike most loss layers, in the TripletLossLayer we can backpropagate
    * * to the first three inputs.
    * */
    virtual inline bool AllowForceBackward(const int bottom_index) const {
      return bottom_index != 3;
    }
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    Blob<Dtype> diff_ap_;  // cached for backward pass
    Blob<Dtype> diff_an_;  // cached for backward pass
    Blob<Dtype> diff_pn_;  // cached for backward pass

    Blob<Dtype> diff_sq_ap_;  // cached for backward pass
    Blob<Dtype> diff_sq_an_;  // tmp storage for gpu forward pass

    Blob<Dtype> dist_sq_ap_;  // cached for backward pass
    Blob<Dtype> dist_sq_an_;  // cached for backward pass

    Blob<Dtype> summer_vec_;  // tmp storage for gpu forward pass
    Blob<Dtype> dist_binary_;  // tmp storage for gpu forward pass

  };

	/**
	* @brief Apply an affine transform on the input layer, where the affine parameters
	*        are learned by back-propagation.
	*        Max Jaderberg etc, Spatial Transformer Networks.
	*
	* TODO(dox): thorough documentation for Forward, Backward, and proto params.
	*/
	template <typename Dtype>
	class TransformerLayer : public Layer<Dtype> {
	public:
		explicit TransformerLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Transformer"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int MinTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int count_;
		Blob<Dtype> CoordinateTarget;//3*(nwh)
		Blob<Dtype> CoordinateSource;//2*(nwh)
		Blob<Dtype> InterpolateWeight;//4*(nwh)
	};
	
	/**
	* @brief Compute covariance matrix for the feature.
	*
	* TODO(dox): thorough documentation for Forward, Backward, and proto params.
	*/
	template <typename Dtype>
	class GramLayer : public Layer<Dtype> {
	public:
    explicit GramLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Gram"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	};

  /**
  * @brief
  */
  template <typename Dtype>
  class EltwiseAffineLayer : public NeuronLayer<Dtype> {
  public:
    /**
    * @param param provides EltwiseAffineParameter EltwiseAffine_param,
    *     with EltwiseAffineLayer options:
    *   - slope_filler (\b optional, FillerParameter,
    *     default {'type': constant 'value':1.0001}).
    *   - bias_filler (\b optional, FillerParameter,
    *     default {'type': constant 'value':0.0001}).
    *   - channel_shared (\b optional, default false).
    *     slopes and biases are shared across channels.
    */
    explicit EltwiseAffineLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "EltwiseAffine"; }

  protected:
    /**
    * @param bottom input Blob vector (length 1)
    *   -# @f$ (N \times C \times ...) @f$
    *      the inputs @f$ x @f$
    * @param top output Blob vector (length 1)
    *   -# @f$ (N \times C \times ...) @f$
    *      the computed outputs for each channel @f$i@f$ @f$
    *        y_i = a_i x_i + b_i
    *      @f$.
    */
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

    /**
    * @brief Computes the error gradient w.r.t. the EltwiseAffine inputs.
    *
    * @param top output Blob vector (length 1), providing the error gradient with
    *      respect to the outputs
    *   -# @f$ (N \times C \times ...) @f$
    *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
    *      with respect to computed outputs @f$ y @f$
    * @param propagate_down see Layer::Backward.
    * @param bottom input Blob vector (length 1)
    *   -# @f$ (N \times C \times ...) @f$
    *      the inputs @f$ x @f$; For each channel @f$i@f$, backward fills their
    *      diff with gradients @f$
    *        \frac{\partial E}{\partial x_i} = \left\{
    *        \begin{array}{lr}
    *            a_i \frac{\partial E}{\partial y_i}
    *        \end{array} \right.
    *      @f$.
    *      If param_propagate_down_[0] is true, it fills the diff with gradients
    *      @f$
    *        \frac{\partial E}{\partial a_i} = \left\{
    *        \begin{array}{lr}
    *            \sum_{x_i} x_i \frac{\partial E}{\partial y_i}
    *        \end{array} \right.
    *      @f$.
    *      If param_propagate_down_[1] is true, it fills the diff with gradients
    *      @f$
    *        \frac{\partial E}{\partial b_i} = \left\{
    *        \begin{array}{lr}
    *             frac{\partial E}{\partial y_i}
    *        \end{array} \right.
    *      @f$.
    */
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    bool channel_shared_;
    Blob<Dtype> multiplier_;  // dot multiplier for backward computation of params
    Blob<Dtype> bias_multiplier_;
    Blob<Dtype> backward_buff_;  // temporary buffer for backward computation
    Blob<Dtype> bottom_memory_;  // memory for in-place computation
  };

  /**
  * @brief Get the sub-region features around some specific points
  *
  * TODO(dox): thorough documentation for Forward, Backward, and proto params.
  */
  template <typename Dtype>
  class SubRegionLayer : public Layer<Dtype> {
  public:
    explicit SubRegionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "SubRegion"; }
    virtual inline int ExactNumBottomBlobs() const { return 3; }
    virtual inline int MinTopBlobs() const { return 1; }

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
    int data_height_;
    int data_width_;
    int as_dim_;
  };

  /**
  * @brief Add noise.
  */
  template <typename Dtype>
  class NoiseLayer : public NeuronLayer<Dtype> {
  public:
    explicit NoiseLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

    virtual inline const char* type() const { return "Noise"; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  };

}  // namespace caffe

#endif  // CAFFE_NEURON_LAYERS_HPP_
