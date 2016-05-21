#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_DEF
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_DEF_ALL
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_DEF_ALL2
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_DEF_ALL3
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_DEF_ALL4
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_LOWRES)
         << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
//  LOG(INFO)<<"this is the end in pool_setup";
  Flag_ = true;
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);// set output mapsize
  //top[0]->NonNeg_ = true;
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }

  kernel_size_ = kernel_h_;
  stride_ = stride_h_;
  pad_ = pad_h_;
  channels_ = bottom[0]->channels();

  if ( (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF) && Flag_ )//set flag in layer_setup
  {
	 // LOG(INFO) << "Deformation layer setup";
	  blobl_a_min_ = this->layer_param_.pooling_param().blobs_a_min();  //read from prototxt, set the mininum of [c1 c2 c3 c4]
      N_ = width_ * height_;   //the input mapsize
      // for deformation layer
      this->blobs_.resize(1); // This blos stores the defw1-defw4
      this->blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, 4));   //for [c1 c2 c3 c4]
      
      Dtype* defw = this->blobs_[0]->mutable_cpu_data();
      LOG(INFO) << "top_buffer_.Reshape" << bottom[0]->num() <<" " << channels_ <<" "<< height_ <<" "<< width_;
      
      top_buffer_.Reshape(bottom[0]->num(), channels_, height_, width_);   //blob data
      LOG(INFO) << "blobl_a_min_:" << blobl_a_min_;
/*       LOG(INFO) << "top_buffer_.mutable_cpu_data";     
      Dtype* top_buffer_data = top_buffer_.mutable_cpu_data();*/
      
      dh_.Reshape(bottom[0]->num(), channels_, 1, 1);   //use to represent the dx
      dv_.Reshape(bottom[0]->num(), channels_, 1, 1);   //dy, which use to calculate the gradient in bp
      Iv_.resize((N_ * channels_*bottom[0]->num() ));     // use to represent the distance in dt
      Ih_.resize((N_ * channels_*bottom[0]->num() ));
      tmpIx_.resize((N_ ));              //use in dt, first calculate horizontal, vertical
      tmpIy_.resize((N_ ));
      defh_.resize((channels_ ));   //   use to represent the part location? yes or no
      defv_.resize((channels_ ));   //
      defp_.resize((channels_ ));  //    h*height + v
      Mdt_.resize((N_));                 //    to save the results in dt
      tmpM_.resize(N_);                 //use in LOW_Res
/*      Iv_.reset(new SyncedMemory(N_ * channels_*bottom[0]->num() * sizeof(int)));
      Ih_.reset(new SyncedMemory(N_ * channels_*bottom[0]->num() * sizeof(int)));
      tmpIx_.reset(new SyncedMemory(N_ * sizeof(int)));
      tmpIy_.reset(new SyncedMemory(N_ * sizeof(int)));
      defh_.reset(new SyncedMemory(channels_ * sizeof(int)));
      defv_.reset(new SyncedMemory(channels_ * sizeof(int)));
      defp_.reset(new SyncedMemory(channels_ * sizeof(int)));
      Mdt_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
      tmpM_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
*/      parts_num_ = 9;     // part
      CHECK_EQ(parts_num_, 9);
      CHECK_EQ(channels_ % parts_num_, 0);
     for (int p = 0; p < parts_num_/3; ++p)
      {
          // 0 1
          // 2 3
          hpos_[p*3] = floor((p*height_)/3) + 1;  // save the part start location
          hpos_[p*3+1] = floor((p*height_)/3) + 1;
          hpos_[p*3+2] = floor((p*height_)/3) + 1;
          vpos_[p*3] = 0 + 1;
          vpos_[p*3+1] = floor((1*width_)/3) + 1;
          vpos_[p*3+2] = floor((2*width_)/3)  + 1;
      }
    /*hpos_[0] = 2;
    hpos_[1] = 2;
    vpos_[0] = 2;
    vpos_[1] = 4;    
    hpos_[2] = 4;
    hpos_[3] = 4;
    vpos_[2] = 2;
    vpos_[3] = 4;*/     
    int* defh = defh_.data();
    int* defv = defv_.data();
    int* defp = defp_.data();
/*    
      int* defh = reinterpret_cast<int*>(defh_->mutable_cpu_data());
      int* defv = reinterpret_cast<int*>(defv_->mutable_cpu_data());
      int* defp = reinterpret_cast<int*>(defp_->mutable_cpu_data());
 */
/*     defw1_.resize(channels_);
      defw2_.resize(channels_);
      defw3_.resize(channels_);
      defw4_.resize(channels_);*/
      for (int c = 0; c < channels_; c++)
      {
          int c1 = c % parts_num_;
          defh[c] = hpos_[c1];   //select one part
          defv[c] = vpos_[c1];
//          LOG(INFO) << "hpos_[" <<c <<"]: " << hpos_[c1]<< " vpos_[" <<c <<"]: " << vpos_[c1] << " width_:" << width_;
          defp[c] = defh[c1] * width_+defv[c1];  // calculate the positon, which will be used in find the proper postion to get data
//          LOG(INFO) << "defp[" <<c1 <<"]: " << defp[c1];
          defw[c*4+0] = blobl_a_min_; // it is same as setting in matlab
          defw[c*4+1] = Dtype(0);
          defw[c*4+2] = blobl_a_min_;
          defw[c*4+3] = Dtype(0);
      }
      LOG(INFO) << "setup defw:" << defw[0*4+0] << ", "<< defw[0*4+1]<< ", " << defw[0*4+2]<< ", " << defw[0*4+3];
      pooled_width_ = 1;
      pooled_height_ = 1;
      top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
              pooled_width_);
  }
  else
  { 
      if (Flag_)
      {   
	  if ( (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF_ALL) || (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF_ALL2) || (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF_ALL3) || (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF_ALL4))
	  {
		  if(this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF_ALL){
			  pooled_height_ = pooled_height_ -1;
			  pooled_width_ = pooled_width_ -1;
		  }


                //  LOG(INFO)<<"this is the start in pool_reshape";
		  int Nparam;
//		  LOG(INFO) << "Deformation layer 2 setup";
		  blobl_a_min_ = this->layer_param_.pooling_param().blobs_a_min();
		  N_ = width_ * height_;  //each map size
		  // for deformation layer
//		  LOG(INFO) << "resize";
		  this->blobs_.resize(1); // This blos stores the defw1-defw4
		  
          if ( (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF_ALL3) || (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF_ALL4) )
			  Nparam = kernel_size_*kernel_size_;     //set kernel size
		  else
			  Nparam = 4;
          
//		  LOG(INFO) << "reset: "<<channels_ <<", " << Nparam;
		  this->blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, Nparam));   //used to represent defw
		  
//		  LOG(INFO) << "Reshape mutable_cpu_data ";
		  Dtype* defw = this->blobs_[0]->mutable_cpu_data();
//		  LOG(INFO) << "top_buffer_.Reshape" << bottom[0]->num() <<" " << channels_ <<" "<< height_ <<" "<< width_;
		  
//		  LOG(INFO) << "Reshape 1 ";
		  top_buffer_.Reshape(bottom[0]->num(), channels_, height_, width_);
//		  LOG(INFO) << "blobl_a_min_:" << blobl_a_min_;
//		  LOG(INFO) << "Nparam:" << Nparam << "channels:" << channels_ << "pooled_height_: " << pooled_height_ << "pooled_width_:" << pooled_width_;
//		  LOG(INFO) << "Reshape 2";
          dh_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
		  dv_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
//		  dh2_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
//		  dv2_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);

        //  LOG(INFO)<<"this is the start 2 in pool_reshape";
          if ( (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF_ALL3) || (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF_ALL4))
          {
//              Iv_.resize((channels_*bottom[0]->num()*pooled_height_*pooled_width_ ));
              Ih_.resize((channels_*bottom[0]->num()*pooled_height_*pooled_width_ ));   //
          }
          else
          {
              Iv_.resize((channels_*bottom[0]->num()*height_*width_ ));   //each point in  inputmap
              Ih_.resize((channels_*bottom[0]->num()*height_*width_ ));
          }
	//  LOG(INFO)<<"this is the start 3 in pool_reshape";
//		  LOG(INFO) << "tmpIx_";
		  tmpIx_.resize((N_ ));  // for one map, n = width * height
		  tmpIy_.resize((N_ ));
		  defh_.resize((channels_ )); // part location
		  defv_.resize((channels_ ));
		  defp_.resize((channels_ ));
		  Mdt_.resize((N_));
		  tmpM_.resize(N_);

		  LOG(INFO) << "setting defw" << " blobl_a_min_:" << blobl_a_min_;
		  if ( (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF_ALL3) || (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF_ALL4) )
			  for (int c = 0; c < channels_; c++)
			  {
				  int defwidx = 0;
				  int center = kernel_size_/2;
				  for (int kh=0; kh<kernel_size_; kh++)
					  for (int kv=0; kv<kernel_size_; kv++)
					  {
                          if (blobl_a_min_ > 0)
                              defw[c*Nparam+defwidx] = -blobl_a_min_* ( (kh-center)*(kh-center) + (kv-center)*(kv-center) );
                          else
                              defw[c*Nparam+defwidx] = 0;
						  defwidx++;
					  }
//				  for (int parami = 0; parami < Nparam; ++parami)
//					  defw[c*Nparam+parami] = Dtype(0);
//		  LOG(INFO) << "defwidx: " <<defwidx;
			  }
		  else
			  for (int c = 0; c < channels_; c++)
			  {
				  /*			  int c1 = c % parts_num_;
				  defh[c] = hpos_[c1];
				  defv[c] = vpos_[c1];
				  //          LOG(INFO) << "hpos_[" <<c <<"]: " << hpos_[c1]<< " vpos_[" <<c <<"]: " << vpos_[c1] << " width_:" << width_;
				  defp[c] = defh[c1] * width_+defv[c1];
				  //          LOG(INFO) << "defp[" <<c1 <<"]: " << defp[c1];*/				  
				  defw[c*4+0] = blobl_a_min_;
				  defw[c*4+1] = Dtype(0);
				  defw[c*4+2] = blobl_a_min_;
				  defw[c*4+3] = Dtype(0);
			  }

		  LOG(INFO) << "setup defw:" << defw[0*4+0] << ", "<< defw[0*4+1]<< ", " << defw[0*4+2]<< ", " << defw[0*4+3];
//		  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
//				  pooled_width_);
		  top[0]->Reshape(bottom[0]->num(), pooled_height_ * pooled_width_ , 1, 1);
	  }
	  else
	  {
		  if ( (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_LOWRES) )
		  {
			  CHECK_EQ(channels_, 3)  << "PoolingLayer LOWRES only takes 3 channels";
			  N_ = width_ * height_;
			  tmpM_.resize(N_*channels_);
			  top[0]->Reshape(bottom[0]->num(), channels_*2+1, pooled_height_,
			  pooled_width_);
		  }
		  else
		  {
			  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
			  pooled_width_);
		  }
	  }
      }
  }
  Flag_ = false;

}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

#define INF 1E20
static inline int square(int x) { return x*x; }

template <typename Dtype>
void max_filter_1d(const Dtype *vals, Dtype *out_vals, int *I, 
                          int step, int n, Dtype a, Dtype b, int s) {
  for (int i = 0; i < n; i++) {
    Dtype max_val = -INF;
    int argmax     = 0;
    int first      = max(0, i-s);
    int last       = min(n-1, i+s);
    for (int j = first; j <= last; j++) {
      Dtype val = *(vals + j*step) - a*square(i-j) - b*(i-j);
      if (val > max_val) {
        max_val = val;
        argmax  = j;
      }
    }
    *(out_vals + i*step) = max_val;
    *(I + i*step) = argmax;
  }
}


template <typename Dtype>
void max_filter_1d(Dtype *vals, Dtype *out_vals, int *I, 
                          int step, int n, Dtype a, Dtype b, int s) {
  for (int i = 0; i < n; i++) {
    Dtype max_val = -INF;
    int argmax     = 0;
    int first      = max(0, i-s);
    int last       = min(n-1, i+s);
    for (int j = first; j <= last; j++) {
      Dtype val = *(vals + j*step) - a*square(i-j) - b*(i-j);
      if (val > max_val) {
        max_val = val;
        argmax  = j;
      }
    }
    *(out_vals + i*step) = max_val;
    *(I + i*step) = argmax;
  }
}

template <typename Dtype>
void dt1d(Dtype *src, Dtype *dst, int *ptr, int step, int n, Dtype a, Dtype b, int * v, Dtype *z) {
  int k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = +INF;
  for (int q = 1; q <= n-1; q++) {
    Dtype s = ((src[q*step] - src[v[k]*step]) - b*(q - v[k]) + a*(square(q) - square(v[k]))) / (2*a*(q-v[k]));
    while (s <= z[k]) {
      // Update pointer
      k--;
      s  = ((src[q*step] - src[v[k]*step]) - b*(q - v[k]) + a*(square(q) - square(v[k]))) / (2*a*(q-v[k]));
    }
    k++;
    v[k]   = q;
    z[k]   = s;
    z[k+1] = +INF;
  }

   k = 0;
   for (int q = 0; q <= n-1; q++) {
     while (z[k+1] < q)
       k++;
     dst[q*step] = a*square(q-v[k]) + b*(q-v[k]) + src[v[k]*step];
     ptr[q*step] = v[k];
  }

}



template <typename Dtype>
void dt1d(const Dtype *src, Dtype *dst, int *ptr, int step, int n, Dtype a, Dtype b, int * v, Dtype *z) {
  int k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = +INF;
  for (int q = 1; q <= n-1; q++) {
    Dtype s = ((src[q*step] - src[v[k]*step]) - b*(q - v[k]) + a*(square(q) - square(v[k]))) / (2*a*(q-v[k]));
    while (s <= z[k]) {
      // Update pointer
      k--;
      s  = ((src[q*step] - src[v[k]*step]) - b*(q - v[k]) + a*(square(q) - square(v[k]))) / (2*a*(q-v[k]));
    }
    k++;
    v[k]   = q;
    z[k]   = s;
    z[k+1] = +INF;
  }

   k = 0;
   for (int q = 0; q <= n-1; q++) {
     while (z[k+1] < q)
       k++;
     dst[q*step] = a*square(q-v[k]) + b*(q-v[k]) + src[v[k]*step];
     ptr[q*step] = v[k];
  }

}




template <typename Dtype>
void PoolingLayer<Dtype>::dt(int dims0, int dims1, const Dtype * vals, Dtype av, Dtype bv, Dtype ah, Dtype bh, int n, int ch) { 

  // Read in deformation coefficients, negating to define a cost
  // Read in offsets for output grid, fixing MATLAB 0-1 indexing
/*  const int *dims = mxGetDimensions(prhs[0]);
  double *vals = (double *)mxGetPr(prhs[0]);
  double ax = mxGetScalar(prhs[1]);
  double bx = mxGetScalar(prhs[2]);
  double ay = mxGetScalar(prhs[3]);
  double by = mxGetScalar(prhs[4]);
  
  mxArray *mxM = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  mxArray *mxIx = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
  mxArray *mxIy = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
  double *M = (double *)mxGetPr(mxM);
  int32_t *Ix_ = (int32_t *)mxGetPr(mxIx);
  int32_t *Iy_ = (int32_t *)mxGetPr(mxIy);

  double *tmpM_ = (double *)mxCalloc(dims[0]*dims[1], sizeof(double));
  int32_t *tmpIx = (int32_t *)mxCalloc(dims[0]*dims[1], sizeof(int32_t));
  int32_t *tmpIy = (int32_t *)mxCalloc(dims[0]*dims[1], sizeof(int32_t));*/
  Dtype* tmpM = tmpM_.data();
  Dtype* Mdt = Mdt_.data();
  int* tmpIy = tmpIy_.data();
  int* tmpIx = tmpIx_.data();
  int* Ih = Ih_.data();
  int* Iv = Iv_.data();
  int maxdim = dims0;
  if (dims0 < dims1)
	  maxdim = dims1;
  int   *vp = new int[maxdim];
  Dtype *zp = new Dtype[maxdim+1];

/*
  Dtype* tmpM =
       reinterpret_cast<Dtype*>(tmpM_->mutable_cpu_data());
  int* tmpIy =
       reinterpret_cast<int*>(tmpIy_->mutable_cpu_data());
  int* tmpIx =
       reinterpret_cast<int*>(tmpIx_->mutable_cpu_data());
  Dtype* Mdt =
       reinterpret_cast<Dtype*>(Mdt_->mutable_cpu_data());
 
  int* Ih = reinterpret_cast<int*>(Ih_->mutable_cpu_data());
  int* Iv = reinterpret_cast<int*>(Iv_->mutable_cpu_data());
*/
  Ih = Ih + (n*channels_+ch)*dims0*dims1;
  Iv = Iv + (n*channels_+ch)*dims0*dims1;
//LOG(INFO) << "dt 1, "<< n <<", " << dims0 <<", " <<dims1 <<", "<<ch;
 // if (n==0)
//	  LOG(INFO) << "dt 1, "<< dims0 <<", " << av <<", " <<bv <<"," << n <<"," <<ch <<"," << Ih[0]<<","<< vals[0];
if (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_DEF)
{
  for (int h = 0; h < dims1; h++)
      //for each vertical location, scan horizontally
    dt1d(vals+h*dims0, tmpM+h*dims0, tmpIy+h*dims0, 1, dims0, -av, -bv, vp, zp);  
  for (int v = 0; v < dims0; v++)
      //for each horizontal location, scan vertically
    dt1d(tmpM+v, Mdt+v, tmpIx+v, dims0, dims1, -ah, -bh, vp, zp);
}
else
{
  for (int h = 0; h < dims1; h++)
      //for each vertical location, scan horizontally
    max_filter_1d(vals+h*dims0, tmpM+h*dims0, tmpIy+h*dims0, 1, dims0, -av, -bv, kernel_size_);  
  for (int v = 0; v < dims0; v++)
      //for each horizontal location, scan vertically
    max_filter_1d(tmpM+v, Mdt+v, tmpIx+v, dims0, dims1, -ah, -bh, kernel_size_);
}
  delete [] vp;
  delete [] zp;
  // get argmins 
//  if (n==0)
//LOG(INFO) << "dt 3"<< "," << Mdt[0] << "," << tmpIx[0];
  for (int h = 0; h < dims1; h++) {
    for (int v = 0; v < dims0; v++) {
      int p = h*dims0+v;
//LOG(INFO) << "dt 3.1:" << h <<", "<< v<<", " << p;
//LOG(INFO) << tmpIx[p];
//LOG(INFO) << Iv[p];
      Ih[p] = tmpIx[p]; // store the best in vertical direction
//LOG(INFO) << "dt 3.2";
      Iv[p] = tmpIy[v+tmpIx[p]*dims0]; // store the best in horizontal direction
    }
  }
//LOG(INFO) << "dt 4";
/*
  mxFree(tmpM);
  mxFree(tmpIx);
  mxFree(tmpIy);
  plhs[0] = mxM;
  plhs[1] = mxIx;
  plhs[2] = mxIy;*/


}



#define	round(x)	((x-floor(x))>0.5 ? ceil(x) : floor(x))

// reduce(im) resizes im to half its size, using a 5-tap binomial filter for anti-aliasing
// (see Burt & Adelson's Laplacian Pyramid paper)

// reduce each column
// result is transposed, so we can apply it twice for a complete reduction
template <typename Dtype>
void PoolingLayer<Dtype>::reduce1dtran(Dtype *src, int sheight, Dtype *dst, int dheight, 
		  int width, int chan) {
  // resize each column of each color channel
  //bzero(dst, chan*width*dheight*sizeof(double));
  memset(dst, 0, chan*width*dheight*sizeof(Dtype));
  int y;
  Dtype *s, *d;

  for (int c = 0; c < chan; c++) {
    for (int x = 0; x < width; x++) {
      s  = src + c*width*sheight + x*sheight;
      d  = dst + c*dheight*width + x;

      // First row
      *d = s[0]*.6875 + s[1]*.2500 + s[2]*.0625;      

      for (y = 1; y < dheight-2; y++) {	
	s += 2;
	d += width;
	*d = s[-2]*0.0625 + s[-1]*.25 + s[0]*.375 + s[1]*.25 + s[2]*.0625;
      }

      // Last two rows
      s += 2;
      d += width;
      if (dheight*2 <= sheight) {
	*d = s[-2]*0.0625 + s[-1]*.25 + s[0]*.375 + s[1]*.25 + s[2]*.0625;
      } else {
	*d = s[1]*.3125 + s[0]*.3750 + s[-1]*.2500 + s[-2]*.0625;
      }
      s += 2;
      d += width;
      *d = s[0]*.6875 + s[-1]*.2500 + s[-2]*.0625;
    }
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::reduce1dtran(const Dtype *src, int sheight, Dtype *dst, int dheight, 
		  int width, int chan) {
  // resize each column of each color channel
  //bzero(dst, chan*width*dheight*sizeof(double));
  memset(dst, 0, chan*width*dheight*sizeof(Dtype));
  int y;
  Dtype *d;
  const Dtype *s;

  for (int c = 0; c < chan; c++) {
    for (int x = 0; x < width; x++) {
      s  = src + c*width*sheight + x*sheight;
      d  = dst + c*dheight*width + x;

      // First row
      *d = s[0]*.6875 + s[1]*.2500 + s[2]*.0625;      

      for (y = 1; y < dheight-2; y++) {	
	s += 2;
	d += width;
	*d = s[-2]*0.0625 + s[-1]*.25 + s[0]*.375 + s[1]*.25 + s[2]*.0625;
      }

      // Last two rows
      s += 2;
      d += width;
      if (dheight*2 <= sheight) {
	*d = s[-2]*0.0625 + s[-1]*.25 + s[0]*.375 + s[1]*.25 + s[2]*.0625;
      } else {
	*d = s[1]*.3125 + s[0]*.3750 + s[-1]*.2500 + s[-2]*.0625;
      }
      s += 2;
      d += width;
      *d = s[0]*.6875 + s[-1]*.2500 + s[-2]*.0625;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
