#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	/**
	* @brief Takes at least two Blob%s and concatenates them along either the num
	*        or channel dimension, outputting the result.
	*/
	template <typename Dtype>
	class BNLayer : public Layer<Dtype> {
	public:
		explicit BNLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "BN"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		// spatial mean & variance
		Blob<Dtype> spatial_statistic_;
		// batch mean & variance
		Blob<Dtype> batch_statistic_;
		// buffer blob
		Blob<Dtype> buffer_blob_;
		// x_norm and x_std
		Blob<Dtype> x_norm_, x_std_;
		// Due to buffer_blob_ and x_norm, this implementation is memory-consuming
		// May use for-loop instead

		// x_sum_multiplier is used to carry out sum using BLAS
		Blob<Dtype> spatial_sum_multiplier_, batch_sum_multiplier_;

		// dimension
		int num_;
		int channels_;
		int height_;
		int width_;
		// eps
		Dtype var_eps_;
		// decay factor
		Dtype decay_;
		// whether or not using moving average for inference
		bool moving_average_;

	};
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
	};

	template <typename Dtype>
	class TripletLossLayer : public LossLayer<Dtype> {
	public:
		explicit TripletLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), diff_() {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline int ExactNumBottomBlobs() const { return 3; }
		virtual inline const char* type() const { return "TripletLoss"; }
		/**
		* Unlike most loss layers, in the TripletLossLayer we can backpropagate
		* to the first three inputs.
		*/
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return bottom_index != 5;
		}

	protected:
		/// @copydoc TripletLossLayer
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		/**
		* @brief Computes the Triplet error gradient w.r.t. the inputs.
		*
		* Computes the gradients with respect to the two input vectors (bottom[0] and
		* bottom[1]), but not the similarity label (bottom[2]).
		*
		* @param top output Blob vector (length 1), providing the error gradient with
		*      respect to the outputs
		*   -# @f$ (1 \times 1 \times 1 \times 1) @f$
		*      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
		*      as @f$ \lambda @f$ is the coefficient of this layer's output
		*      @f$\ell_i@f$ in the overall Net loss
		*      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
		*      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
		*      (*Assuming that this top Blob is not used as a bottom (input) by any
		*      other layer of the Net.)
		* @param propagate_down see Layer::Backward.
		* @param bottom input Blob vector (length 2)
		*   -# @f$ (N \times C \times 1 \times 1) @f$
		*      the features @f$a@f$; Backward fills their diff with
		*      gradients if propagate_down[0]
		*   -# @f$ (N \times C \times 1 \times 1) @f$
		*      the features @f$b@f$; Backward fills their diff with gradients if
		*      propagate_down[1]
		*/
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> diff_;  // cached for backward pass
		Blob<Dtype> diff_pos;
		Blob<Dtype> diff_neg;
		Blob<Dtype> diff_par;
		Blob<Dtype> dist_sq_;  // cached for backward pass
		Blob<Dtype> dist_sq_pos;
		Blob<Dtype> dist_sq_neg;
		Blob<Dtype> dist_sq_par;
		Blob<Dtype> diff_sq_;  // tmp storage for gpu forward pass
		Blob<Dtype> diff_sq_pos;
		Blob<Dtype> diff_sq_neg;
		Blob<Dtype> diff_sq_par;
		Blob<Dtype> summer_vec_;  // tmp storage for gpu forward pass
	};

	/**
	* @brief Normalizes input.
	*/
	template <typename Dtype>
	class NormalizeLayer : public Layer<Dtype> {
	public:
		explicit NormalizeLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Normalize"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> sum_multiplier_, norm_, squared_;
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
	class CovarianceLayer : public Layer<Dtype> {
	public:
		explicit CovarianceLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Covariance"; }
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

}  // namespace caffe

#endif  // CAFFE_NEURON_LAYERS_HPP_
