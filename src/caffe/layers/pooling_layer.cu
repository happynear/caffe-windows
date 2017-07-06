#include <algorithm>
#include <cfloat>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const rand_idx, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = 0.;
    Dtype cumvalues = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = (cumsum > 0.) ? cumvalues / cumsum : 0.;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

//  LOG(INFO)<<"the entry in forward";
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  Dtype* bottom_diff;
  Dtype* dh;
  Dtype* dv;
  Dtype* Mdt;
 //   int N_;
  int* defh;
  int* defv;
  int* defp;
  int* Ih;
  int* Iv;
  Dtype* defw;
  int Nparam = kernel_size_*kernel_size_;
  const Dtype* bottom_data_p;

  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_DEF:
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 1";
//    top_buffer_data = top_buffer_.mutable_cpu_data();  

    bottom_diff = (bottom)[0]->mutable_cpu_diff();
	memset(bottom_diff, 0, top[0]->num()*channels_*width_*height_*sizeof(Dtype));
	bottom_diff=(bottom)[0]->mutable_gpu_diff();

    dh_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
    dv_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
    Iv_.resize((channels_*bottom[0]->num()*height_*width_ ));   //each point in  inputmap
    Ih_.resize((channels_*bottom[0]->num()*height_*width_ ));

    top[0]->Reshape(bottom[0]->num(), channels_, 1, 1); 
	bottom_data_p = bottom[0]->cpu_data();
    top_data = top[0]->mutable_cpu_data();
    Mdt = Mdt_.data();
//    Mdt = reinterpret_cast<Dtype*>(Mdt_->mutable_cpu_data());
//    caffe_copy(channels_*height_*width_, bottom_data, top_buffer_data);

    dh = dh_.mutable_cpu_data();
    dv = dv_.mutable_cpu_data();
    defw = this->blobs_[0]->mutable_cpu_data();

    for (int ch = 0; ch < channels_; ++ch) {
		
        if (defw[ch*4+0] < blobl_a_min_)
            defw[ch*4+0] = blobl_a_min_;
        if (defw[ch*4+2] < blobl_a_min_)
            defw[ch*4+2] = blobl_a_min_;        

/*		for (int i=0; i< 4; ++i)
			if (fabs(defw[ch*4+i]) > 5)
				LOG(INFO) << "defw > 5" <<ch <<": " << defw[ch*4+i];*/
//if (  (ch==0) )
//    LOG(INFO) << "defw" <<ch <<": " << defw[ch*4+0] << ", "<< defw[ch*4+1]<< ", " << defw[ch*4+2]<< ", " << defw[ch*4+3];
    }
//    LOG(INFO) << "ff defw:" << defw[0*4+0] << ", "<< defw[0*4+1]<< ", " << defw[0*4+2]<< ", " << defw[0*4+3];
//		  bool Flag_print;

    N_ = width_ * height_;
    defh = defh_.data();
    defv = defv_.data();
    defp = defp_.data();
    Ih = Ih_.data();
    Iv = Iv_.data();
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2"<<bottom[0]->num() <<","<<channels_;
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int ch = 0; ch < channels_; ++ch) {          
          const Dtype* data_pointer = bottom_data_p + bottom[0]->offset(n, ch);
		 /* Flag_print = 1;
		  for (int h =0; h < height_; h++)
		  for (int v =0; (v < width_)&&Flag_print; v++)
			  if (data_pointer[h*width_+v] > 100)
			  {
				  LOG(INFO) << "data_pointer["<< h <<"][" <<v <<"] > 100," <<data_pointer[h*width_+v] <<"," <<n <<"," << ch;
				  //Flag_print = false;
			  }*/
				  
//          Dtype* data_pointer2 = top_buffer_data  + top_buffer_.offset(n, ch);
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.1 "<<"ch="<< ch ; //<<"ch="<< ch <<" offset:" << top_buffer_.offset(0, ch) << " " << data_pointer[0];
//if ( (n==0) && data_pointer[0] > 1)
//	LOG(INFO) << "Forward_gpu dt1:" <<"ch="<< ch <<" offset:" << top_buffer_.offset(n, ch) << "vals: " << data_pointer[0] <<", " << data_pointer[4]<<"," << height_ << "," << width_;

          dt(width_, height_, data_pointer, defw[ch*4+0], defw[ch*4+1], defw[ch*4+2], defw[ch*4+3], n, ch);
          //LOG(INFO) << "defw" <<ch <<": " << defw[ch*4+0] << ", "<< defw[ch*4+1]<< ", " << defw[ch*4+2]<< ", " << defw[ch*4+3];
//LOG(INFO) << "ff defw:" << defw[ch*4+0] << ", "<< defw[ch*4+1]<< ", " << defw[ch*4+2]<< ", " << defw[ch*4+3];
          // obtain the score of each part in each channel
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.2";
//LOG(INFO) << defp[ch]<<","<<data_pointer[defp[ch]]<<","<<Mdt[defp[ch]];
//LOG(INFO) << n*channels_+ch;
//if (Mdt[defp[ch]] > 100)
//	LOG(INFO) << "fp data:" << defp[ch] << "," << data_pointer[defp[ch]] <<","<< data_pointer[defp[ch]-1] <<"," << data_pointer[defp[ch]+1] << ", "<< Mdt[defp[ch]];
          top_data[n*channels_+ch] = Mdt[defp[ch]];
          
/*          // facillitate BP using conv
LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.3";
          memset(data_pointer2, 0, sizeof(Dtype) * N_);
LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.4";
          data_pointer2[defp[ch]] = top_data[n*channels_+ch];*/
//LOG(INFO) << "dh_dv" <<ch <<": "<<defh[ch] << ", "<< Ih[(n*channels_+ch)*N_+defp[ch]]<< ", " << defv[ch] << ", "<<Iv[(n*channels_+ch)*N_+defp[ch]];
if ((n==0)&&( top_data[n*channels_+ch] > 10000))
	LOG(INFO) << "Forward_gpu dt2!!: vals:" << top_data[n*channels_+ch] <<" from " << data_pointer[defp[ch]];
          
          // obtain the dh and dv of each part
          dh[n*channels_+ch] = defh[ch] - Ih[(n*channels_+ch)*N_+defp[ch]];
          dv[n*channels_+ch] = defv[ch] - Iv[(n*channels_+ch)*N_+defp[ch]];
//CHECK_EQ(Flag_print, 1);

//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.5";
      }
    }
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.6:" << top_data[0];
    top[0]->mutable_gpu_data();
//    bottom[0]->mutable_gpu_data();
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 3";
    break;
  case PoolingParameter_PoolMethod_DEF_ALL:
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 1";
//    top_buffer_data = top_buffer_.mutable_cpu_data();  
//    LOG(INFO)<<"the entry in DEF_ALL forward";
    bottom_diff = (bottom)[0]->mutable_cpu_diff();
	memset(bottom_diff, 0, top[0]->num()*channels_*width_*height_*sizeof(Dtype));
	bottom_diff=(bottom)[0]->mutable_gpu_diff();

    dh_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
    dv_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
    Iv_.resize((channels_*bottom[0]->num()*height_*width_ ));   //each point in  inputmap
    Ih_.resize((channels_*bottom[0]->num()*height_*width_ ));
    //tmpIx_.resize((N_ ));  // for one map, n = width * height
    //tmpIy_.resize((N_ ));
    //defh_.resize((channels_ )); // part location
    //defv_.resize((channels_ ));
    //defp_.resize((channels_ ));
    //Mdt_.resize((N_));
    //tmpM_.resize(N_); 
    top[0]->Reshape(bottom[0]->num(), pooled_height_ * pooled_width_, 1, 1);   

    bottom_data_p = bottom[0]->cpu_data();
    top_data = top[0]->mutable_cpu_data();
    Mdt = Mdt_.data();
//    Mdt = reinterpret_cast<Dtype*>(Mdt_->mutable_cpu_data());
//    caffe_copy(channels_*height_*width_, bottom_data, top_buffer_data);

    dh = dh_.mutable_cpu_data();
    dv = dv_.mutable_cpu_data();
    memset(dh, 0, top[0]->num()*channels_*width_*height_*sizeof(Dtype));
    memset(dv, 0, top[0]->num()*channels_*width_*height_*sizeof(Dtype));
    defw = this->blobs_[0]->mutable_cpu_data();

    for (int ch = 0; ch < channels_; ++ch) {		
        if (defw[ch*4+0] < blobl_a_min_)
            defw[ch*4+0] = blobl_a_min_;
        if (defw[ch*4+2] < blobl_a_min_)
            defw[ch*4+2] = blobl_a_min_;       
    }

    N_ = width_ * height_;
    defh = defh_.data();  // for part horizantal start point
    defv = defv_.data();  //for vertical start point
    defp = defp_.data();  //the w*dim0 + h
    Ih = Ih_.data();      //save the best location for dt
    Iv = Iv_.data();
//    LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2 "<<bottom[0]->num() <<","<<channels_<<","<<width_<<","<<height_;
//	LOG(INFO) << "fp offset:" << bottom[0]->offset(0, 1);
    for (int n = 0; n < bottom[0]->num(); ++n) {
	  int N2 = pooled_height_*pooled_width_;
      for (int ch = 0; ch < channels_; ++ch) {          
          const Dtype* data_pointer = bottom_data_p + bottom[0]->offset(n, ch);
		  /* Flag_print = 1;
		  for (int h =0; h < height_; h++)
		  for (int v =0; (v < width_)&&Flag_print; v++)
			  if (data_pointer[h*width_+v] > 100)
			  {
				  LOG(INFO) << "data_pointer["<< h <<"][" <<v <<"] > 100," <<data_pointer[h*width_+v] <<"," <<n <<"," << ch;
				  Flag_print = false;
			  }*/
				  
//          Dtype* data_pointer2 = top_buffer_data  + top_buffer_.offset(n, ch);
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.1 "<<"ch="<< ch ; //<<"ch="<< ch <<" offset:" << top_buffer_.offset(0, ch) << " " << data_pointer[0];
//if ( (n==0) && data_pointer[0] > 1)

		  //LOG(INFO) << "Forward_gpu dt1:" <<"ch="<< ch <<" offset:" << top_buffer_.offset(n, ch) << "vals: " << data_pointer[0] <<", " << data_pointer[4]<<"," << height_ << "," << width_;
		 /* for (int phS=0; phS<height_; ++phS)
			  for (int pwS = 0; pwS < width_; ++pwS) {
				  int ihval = Ih[(n*channels_+ch)*N_+phS * width_ + pwS];
				  int ivval = Iv[(n*channels_+ch)*N_+phS * width_ + pwS];
				  LOG(INFO) << "fp data4[" << phS << "," << pwS <<"]:" << data_pointer[phS * width_ + pwS];
			  }*/
          dt(width_, height_, data_pointer, defw[ch*4+0], defw[ch*4+1], defw[ch*4+2], defw[ch*4+3], n, ch);
		  /*for (int phS=0; phS<height_; ++phS)
			  for (int pwS = 0; pwS < width_; ++pwS) {
				  int ihval = Ih[(n*channels_+ch)*N_+phS * width_ + pwS];
				  int ivval = Iv[(n*channels_+ch)*N_+phS * width_ + pwS];
				  LOG(INFO) << data_pointer[phS * width_ + pwS] ;
				  LOG(INFO) << "fp data3[" << phS << "," << pwS <<"]:" << "=>" << "[" << ihval << "," << ivval << "]:" << Mdt[phS * width_ + pwS];
			  }*/

          // obtain the score of each part in each channel
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.2"<<","<<n<<","<<ch;
//LOG(INFO) << defp[ch];
//LOG(INFO) << n*channels_+ch;
//if (Mdt[defp[ch]] > 100)
//	LOG(INFO) << "fp data:" << defp[ch] << "," << data_pointer[defp[ch]] <<","<< data_pointer[defp[ch]-1] <<"," << data_pointer[defp[ch]+1] << ", "<< Mdt[defp[ch]];

//		  dh[n*channels_+ch] = 0;
//		  dv[n*channels_+ch] = 0;
//		  LOG(INFO)<< "defw:" << defw[ch*4+0] <<"," << defw[ch*4+1] <<","<< defw[ch*4+2]<<","<< defw[ch*4+3];
		//  for (int ph = 0; ph < pooled_height_; ++ph) {
			  int ph = static_cast<int>(ch/pooled_width_);
			  int pw = ch - ph*pooled_width_;
			  int phS = ph * stride_;
		//	 for (int pw = 0; pw < pooled_width_; ++pw) {
				  int pwS = pw * stride_;
				  int defval = phS * width_ + pwS;
				 // LOG(INFO) << ph <<","<< pw;
				  int ihval = Ih[(n*channels_+ch)*N_+phS * width_ + pwS];
				  int ivval = Iv[(n*channels_+ch)*N_+phS * width_ + pwS];
				  Dtype hdif = phS - Ih[(n*channels_+ch)*N_+phS * width_ + pwS];
				  Dtype vdif = pwS - Iv[(n*channels_+ch)*N_+phS * width_ + pwS];
				//  LOG(INFO) << "fp data[" << defval << "]:" << data_pointer[defval] <<","<< Mdt[defval];
				//  LOG(INFO) << "fp data2:" << phS <<"," <<pwS << "=> " << ihval << "," << ivval << "," << data_pointer[ihval * width_ + ivval];
				//  top_data[ph * pooled_width_ + pw] = Mdt[phS * width_ + pwS];
				  top_data[0] = Mdt[phS * width_ + pwS];
				  dh[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = hdif;
				  dv[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = vdif;
		//	  }
		// }
        top_data += top[0]->offset(0, 1);
//          top_data[n*channels_+ch] = Mdt[defp[ch]];

/*          // facillitate BP using conv
LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.3";
          memset(data_pointer2, 0, sizeof(Dtype) * N_);
LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.4";
          data_pointer2[defp[ch]] = top_data[n*channels_+ch];*/
//LOG(INFO) << "dh_dv" <<ch <<": "<<defh[ch] << ", "<< Ih[(n*channels_+ch)*N_+defp[ch]]<< ", " << defv[ch] << ", "<<Iv[(n*channels_+ch)*N_+defp[ch]];
//		if ((n==0)&&( top_data[n*channels_+ch] > 10000))
//			LOG(INFO) << "Forward_gpu dt2!!: vals:" << top_data[n*channels_+ch] <<" from " << data_pointer[defp[ch]];
          
          // obtain the dh and dv of each part
//CHECK_EQ(Flag_print, 1);

//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.5";
      }
    }
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.6:" << top_data[0];
    top[0]->mutable_gpu_data();
//    LOG(INFO)<<"the end in DEF_ALL forward";
//    bottom[0]->mutable_gpu_data();
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 3";
    break;
  case PoolingParameter_PoolMethod_DEF_ALL2:
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 1";
//    top_buffer_data = top_buffer_.mutable_cpu_data();  

    bottom_diff = (bottom)[0]->mutable_cpu_diff();
	memset(bottom_diff, 0, top[0]->num()*channels_*width_*height_*sizeof(Dtype));
	bottom_diff=(bottom)[0]->mutable_gpu_diff();

	bottom_data_p = bottom[0]->cpu_data();
    top_data = top[0]->mutable_cpu_data();
    Mdt = Mdt_.data();
//    Mdt = reinterpret_cast<Dtype*>(Mdt_->mutable_cpu_data());
//    caffe_copy(channels_*height_*width_, bottom_data, top_buffer_data);

    dh = dh_.mutable_cpu_data();
    dv = dv_.mutable_cpu_data();
    defw = this->blobs_[0]->mutable_cpu_data();

    for (int ch = 0; ch < channels_; ++ch) {		
        if (defw[ch*4+0] < blobl_a_min_)
            defw[ch*4+0] = blobl_a_min_;
        if (defw[ch*4+2] < blobl_a_min_)
            defw[ch*4+2] = blobl_a_min_;       
    }

    N_ = width_ * height_;
    defh = defh_.data();
    defv = defv_.data();
    defp = defp_.data();
    Ih = Ih_.data();
    Iv = Iv_.data();
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2"<<bottom[0]->num() <<","<<channels_;
//	LOG(INFO) << "fp offset:" << bottom[0]->offset(0, 1);
    for (int n = 0; n < bottom[0]->num(); ++n) {
	  int N2 = pooled_height_*pooled_width_;
      for (int ch = 0; ch < channels_; ++ch) {          
          const Dtype* data_pointer = bottom_data_p + bottom[0]->offset(n, ch);
/*		  Flag_print = 1;
		  for (int h =0; h < height_; h++)
		  for (int v =0; (v < width_)&&Flag_print; v++)
			  if (data_pointer[h*width_+v] > 100)
			  {
				  LOG(INFO) << "data_pointer["<< h <<"][" <<v <<"] > 100," <<data_pointer[h*width_+v] <<"," <<n <<"," << ch;
				  Flag_print = false;
			  }*/
				  
//          Dtype* data_pointer2 = top_buffer_data  + top_buffer_.offset(n, ch);
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.1 "<<"ch="<< ch ; //<<"ch="<< ch <<" offset:" << top_buffer_.offset(0, ch) << " " << data_pointer[0];
//if ( (n==0) && data_pointer[0] > 1)

/*		  LOG(INFO) << "Forward_gpu dt1:" <<"ch="<< ch <<" offset:" << top_buffer_.offset(n, ch) << "vals: " << data_pointer[0] <<", " << data_pointer[4]<<"," << height_ << "," << width_;
		  for (int phS=0; phS<height_; ++phS)
			  for (int pwS = 0; pwS < width_; ++pwS) {
				  int ihval = Ih[(n*channels_+ch)*N_+phS * width_ + pwS];
				  int ivval = Iv[(n*channels_+ch)*N_+phS * width_ + pwS];
				  LOG(INFO) << "fp data4[" << phS << "," << pwS <<"]:" << data_pointer[phS * width_ + pwS];
			  }*/
          dt(width_, height_, data_pointer, defw[ch*4+0], defw[ch*4+1], defw[ch*4+2], defw[ch*4+3], n, ch);
/*		  for (int phS=0; phS<height_; ++phS)
			  for (int pwS = 0; pwS < width_; ++pwS) {
				  int ihval = Ih[(n*channels_+ch)*N_+phS * width_ + pwS];
				  int ivval = Iv[(n*channels_+ch)*N_+phS * width_ + pwS];
				  LOG(INFO) << "fp data3[" << phS << "," << pwS <<"]:" << data_pointer[phS * width_ + pwS] << "=>" << "[" << ihval << "," << ivval << "]:" << Mdt[phS * width_ + pwS];
			  }
*/
          // obtain the score of each part in each channel
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.2";
//LOG(INFO) << defp[ch];
//LOG(INFO) << n*channels_+ch;
//if (Mdt[defp[ch]] > 100)
//	LOG(INFO) << "fp data:" << defp[ch] << "," << data_pointer[defp[ch]] <<","<< data_pointer[defp[ch]-1] <<"," << data_pointer[defp[ch]+1] << ", "<< Mdt[defp[ch]];

//		  dh[n*channels_+ch] = 0;
//		  dv[n*channels_+ch] = 0;
//		  LOG(INFO)<< "defw:" << defw[ch*4+0] <<"," << defw[ch*4+1] <<","<< defw[ch*4+2]<<","<< defw[ch*4+3];
		  for (int ph = 0; ph < pooled_height_; ++ph) {
			  int phS = ph * stride_;
			  for (int pw = 0; pw < pooled_width_; ++pw) {
				  int pwS = pw * stride_;
				  int defval = phS * width_ + pwS;
				  int ihval = Ih[(n*channels_+ch)*N_+phS * width_ + pwS];
				  int ivval = Iv[(n*channels_+ch)*N_+phS * width_ + pwS];
				  Dtype hdif = phS - Ih[(n*channels_+ch)*N_+phS * width_ + pwS];
				  Dtype vdif = pwS - Iv[(n*channels_+ch)*N_+phS * width_ + pwS];
//				  LOG(INFO) << "fp data[" << defval << "]:" << data_pointer[defval] <<","<< data_pointer[defval-1] <<"," << data_pointer[defval+1] << ", "<< Mdt[defval];
//				  LOG(INFO) << "fp data2:" << phS <<"," <<pwS << "=> " << ihval << "," << ivval << "," << data_pointer[ihval * width_ + ivval];
				  top_data[ph * pooled_width_ + pw] = data_pointer[ihval * width_ + ivval];
				  dh[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = hdif;
				  dv[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = vdif;
			  }
		  }
        top_data += top[0]->offset(0, 1);
//          top_data[n*channels_+ch] = Mdt[defp[ch]];

/*          // facillitate BP using conv
LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.3";
          memset(data_pointer2, 0, sizeof(Dtype) * N_);
LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.4";
          data_pointer2[defp[ch]] = top_data[n*channels_+ch];*/
//LOG(INFO) << "dh_dv" <<ch <<": "<<defh[ch] << ", "<< Ih[(n*channels_+ch)*N_+defp[ch]]<< ", " << defv[ch] << ", "<<Iv[(n*channels_+ch)*N_+defp[ch]];
//		if ((n==0)&&( top_data[n*channels_+ch] > 10000))
//			LOG(INFO) << "Forward_gpu dt2!!: vals:" << top_data[n*channels_+ch] <<" from " << data_pointer[defp[ch]];
          
          // obtain the dh and dv of each part
//CHECK_EQ(Flag_print, 1);

//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.5";
      }
    }
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.6:" << top_data[0];
    top[0]->mutable_gpu_data();
    break;
  case PoolingParameter_PoolMethod_DEF_ALL3:
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF_ALL3";
//    top_buffer_data = top_buffer_.mutable_cpu_data();  

    bottom_diff = (bottom)[0]->mutable_cpu_diff();
	memset(bottom_diff, 0, top[0]->num()*channels_*width_*height_*sizeof(Dtype));
	bottom_diff=(bottom)[0]->mutable_gpu_diff();

	bottom_data_p = bottom[0]->cpu_data();
    top_data = top[0]->mutable_cpu_data();
    Mdt = Mdt_.data();
//    Mdt = reinterpret_cast<Dtype*>(Mdt_->mutable_cpu_data());
//    caffe_copy(channels_*height_*width_, bottom_data, top_buffer_data);

    dh = dh_.mutable_cpu_data();
    dv = dv_.mutable_cpu_data();
    defw = this->blobs_[0]->mutable_cpu_data();

    N_ = width_ * height_;
    defh = defh_.data();
    defv = defv_.data();
    defp = defp_.data();
    Ih = Ih_.data();
    Iv = Iv_.data();
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2"<<bottom[0]->num() <<","<<channels_;
//	LOG(INFO) << "fp offset:" << bottom[0]->offset(0, 1);

/*       for (int ch = 0; ch < channels_; ++ch) {          
		  for (int ph = 0; ph < pooled_height_; ++ph) {
			  for (int pw = 0; pw < pooled_width_; ++pw) {
                  int idx = ch*Nparam+ph*pooled_height_+pw;
                  if (defw[idx] < 0)
                      defw[idx] = 0;
              }
          }
      }*/
    
    for (int n = 0; n < bottom[0]->num(); ++n) {
       
	  int N2 = pooled_height_*pooled_width_;
	  int center = kernel_size_/2;
	  int def_center = center+center*kernel_size_;
      

      for (int ch = 0; ch < channels_; ++ch) {          
          const Dtype* data_pointer = bottom_data_p + bottom[0]->offset(n, ch);
/*		  Flag_print = 1;
		  for (int h =0; h < height_; h++)
		  for (int v =0; (v < width_)&&Flag_print; v++)
			  if (data_pointer[h*width_+v] > 100)
			  {
				  LOG(INFO) << "data_pointer["<< h <<"][" <<v <<"] > 100," <<data_pointer[h*width_+v] <<"," <<n <<"," << ch;
				  Flag_print = false;
			  }*/
				  
//          Dtype* data_pointer2 = top_buffer_data  + top_buffer_.offset(n, ch);
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.1 "<<"ch="<< ch ; //<<"ch="<< ch <<" offset:" << top_buffer_.offset(0, ch) << " " << data_pointer[0];
//if ( (n==0) && data_pointer[0] > 1)

/*		  LOG(INFO) << "Forward_gpu dt1:" <<"ch="<< ch <<" offset:" << top_buffer_.offset(n, ch) << "vals: " << data_pointer[0] <<", " << data_pointer[4]<<"," << height_ << "," << width_;
		  for (int phS=0; phS<height_; ++phS)
			  for (int pwS = 0; pwS < width_; ++pwS) {
				  int ihval = Ih[(n*channels_+ch)*N_+phS * width_ + pwS];
				  int ivval = Iv[(n*channels_+ch)*N_+phS * width_ + pwS];
				  LOG(INFO) << "fp data4[" << phS << "," << pwS <<"]:" << data_pointer[phS * width_ + pwS];
			  }*/
//          dt(width_, height_, data_pointer, defw[ch*4+0], defw[ch*4+1], defw[ch*4+2], defw[ch*4+3], n, ch);
/*		  for (int phS=0; phS<height_; ++phS)
			  for (int pwS = 0; pwS < width_; ++pwS) {
				  int ihval = Ih[(n*channels_+ch)*N_+phS * width_ + pwS];
				  int ivval = Iv[(n*channels_+ch)*N_+phS * width_ + pwS];
				  LOG(INFO) << "fp data3[" << phS << "," << pwS <<"]:" << data_pointer[phS * width_ + pwS] << "=>" << "[" << ihval << "," << ivval << "]:" << Mdt[phS * width_ + pwS];
			  }
*/
          // obtain the score of each part in each channel
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2.2";
//LOG(INFO) << defp[ch];
//LOG(INFO) << n*channels_+ch;
//if (Mdt[defp[ch]] > 100)
//	LOG(INFO) << "fp data:" << defp[ch] << "," << data_pointer[defp[ch]] <<","<< data_pointer[defp[ch]-1] <<"," << data_pointer[defp[ch]+1] << ", "<< Mdt[defp[ch]];

//		  dh[n*channels_+ch] = 0;
//		  dv[n*channels_+ch] = 0;
//		  LOG(INFO)<< "defw:" << defw[ch*4+0] <<"," << defw[ch*4+1] <<","<< defw[ch*4+2]<<","<< defw[ch*4+3];
		  for (int ph = 0; ph < pooled_height_; ++ph) {
			  int phS = ph * stride_;
			  if (phS > height_-kernel_size_)
				  phS = height_ - kernel_size_;
			  for (int pw = 0; pw < pooled_width_; ++pw) {
                  int pwS = pw * stride_;
				  int defval = phS * width_ + pwS;
				  int defwidx;
				  int maxIdx;
                  
                  int hstart = ph * stride_h_ - pad_h_;
                  int wstart = pw * stride_w_ - pad_w_;
                  int hstart_ori = hstart;
                  int wstart_ori = wstart;
                  int hend = min(hstart + kernel_h_, height_);
                  int wend = min(wstart + kernel_w_, width_);
                  hstart = max(hstart, 0);
                  wstart = max(wstart, 0);
                  const int pool_index = ph * pooled_width_ + pw;
                  Dtype maxVal = -FLT_MAX;
                  for (int h = hstart; h < hend; ++h) {
                      for (int w = wstart; w < wend; ++w) {
                          const int index = h * width_ + w;
                          int kh = h-hstart_ori;
                          int kv = w-wstart_ori;
                          int defwidx = kh*kernel_size_+kv;
						  Dtype CurVal1 = data_pointer[index];
						  Dtype CurVal2 = CurVal1+defw[ch*Nparam+defwidx];
                          if (CurVal2 > maxVal) {
                              maxVal = CurVal2;
//                              mask[pool_index] = index;
                              maxIdx = defwidx; 
							  Ih[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = index;
                          }
                      }
                  }
				  top_data[pool_index] = maxVal;
				  dh[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = maxIdx;
/*
                  int pwS = pw * stride_;
				  int defval = phS * width_ + pwS;
				  int defwidx;
				  int maxIdx;
				  if (pwS > width_-kernel_size_)
					  pwS = width_ - kernel_size_;
				  maxIdx = def_center;
				  Dtype maxVal = -FLT_MAX;
				  defwidx = 0;
				  for (int kh=0; kh<kernel_size_; kh++)
					  for (int kv=0; kv<kernel_size_; kv++)
					  {
						  Dtype CurVal1 = data_pointer[(phS+kh) * width_ + pwS+kv];
						  Dtype CurVal2 = CurVal1+defw[ch*Nparam+defwidx];
                          if (fabs(CurVal1-CurVal2) > 0.0001)
                                  LOG(INFO) << "CurVal1, 2:" << CurVal1 <<", "<< CurVal2;
						  if (maxVal < CurVal2)
						  {
							  maxVal = CurVal2;
							  maxIdx = defwidx;
							  Ih[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = (phS+kh) * width_ + pwS+kv;
						  }							  
						  defwidx++;
					  }
				  top_data[ph * pooled_width_ + pw] = maxVal;
				  dh[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = maxIdx;
//				  dv[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = vdif;
 */
			  }
		  }
		  top_data += top[0]->offset(0, 1);          
      }
    }
    top[0]->mutable_gpu_data();
    break;
  case PoolingParameter_PoolMethod_DEF_ALL4:
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 1";
//    top_buffer_data = top_buffer_.mutable_cpu_data();  

    bottom_diff = (bottom)[0]->mutable_cpu_diff();
	memset(bottom_diff, 0, top[0]->num()*channels_*width_*height_*sizeof(Dtype));
	bottom_diff=(bottom)[0]->mutable_gpu_diff();

	bottom_data_p = bottom[0]->cpu_data();
    top_data = top[0]->mutable_cpu_data();
    Mdt = Mdt_.data();
//    Mdt = reinterpret_cast<Dtype*>(Mdt_->mutable_cpu_data());
//    caffe_copy(channels_*height_*width_, bottom_data, top_buffer_data);

    dh = dh_.mutable_cpu_data();
    dv = dv_.mutable_cpu_data();
    defw = this->blobs_[0]->mutable_cpu_data();

    N_ = width_ * height_;
    defh = defh_.data();
    defv = defv_.data();
    defp = defp_.data();
    Ih = Ih_.data();
    Iv = Iv_.data();
//LOG(INFO) << "Forward_gpu PoolingParameter_PoolMethod_DEF 2"<<bottom[0]->num() <<","<<channels_;
//	LOG(INFO) << "fp offset:" << bottom[0]->offset(0, 1);
    for (int n = 0; n < bottom[0]->num(); ++n) {
	  int N2 = pooled_height_*pooled_width_;
      for (int ch = 0; ch < channels_; ++ch) {          
          const Dtype* data_pointer = bottom_data_p + bottom[0]->offset(n, ch);
		  for (int ph = 0; ph < pooled_height_; ++ph) {
			  int phS = ph * stride_;
			  if (phS > height_-kernel_size_)
				  phS = height_ - kernel_size_;
			  for (int pw = 0; pw < pooled_width_; ++pw) {
				  int pwS = pw * stride_;
				  int defval = phS * width_ + pwS;
				  int defwidx;
				  int maxIdx;
				  maxIdx = 0;
				  if (pwS > width_-kernel_size_)
					  pwS = width_ - kernel_size_;
				  Dtype maxVal1 = data_pointer[phS * width_ + pwS];
				  Dtype maxVal = maxVal1+defw[ch*Nparam+0];
				  Ih[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = defval;
				  defwidx = 0;
				  for (int kh=0; kh<kernel_size_; kh++)
					  for (int kv=0; kv<kernel_size_; kv++)
					  {
						  Dtype CurVal1 = data_pointer[(phS+kh) * width_ + pwS+kv];
						  Dtype CurVal2 = CurVal1+defw[ch*Nparam+defwidx];
						  if (maxVal < CurVal2)
						  {
							  maxVal = CurVal2;
							  maxVal1 = CurVal1;
							  maxIdx = defwidx;
							  Ih[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = (phS+kh) * width_ + pwS+kv;
						  }							  
						  defwidx++;
					  }
				  top_data[ph * pooled_width_ + pw] = maxVal1;
				  dh[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = maxIdx;
//				  dv[(n*channels_+ch)*N2+ph * pooled_width_ + pw] = vdif;
			  }
		  }
		  top_data += top[0]->offset(0, 1);          
      }
    }
    top[0]->mutable_gpu_data();
    break;
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* const rand_idx, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const rand_idx_slice =
        rand_idx + (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx_slice[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //LOG(INFO)<<"the entry in backward";
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;

  Dtype* d_defw;
  Dtype* dh;
  Dtype* dv;
  int* Iv;
  int* Ih;
  int* defp;
  int N2;
  int Nparam = kernel_size_*kernel_size_;
  const Dtype* bottom_data_p;

  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_DEF:
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF";
/*  top_data = top[0]->cpu_data();
  bottom_data = (*bottom)[0]->cpu_data();*/
    bottom_diff = bottom[0]->mutable_cpu_diff();
    top_diff = top[0]->cpu_diff();

   // LOG(INFO)<<"top_channels:"<<top[0]->channels();
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 1";
    Ih = Ih_.data();
    Iv = Iv_.data();
    defp = defp_.data();
//    Ih = reinterpret_cast<int*>(Ih_->mutable_cpu_data());
//    Iv = reinterpret_cast<int*>(Iv_->mutable_cpu_data());
//    defp = reinterpret_cast<int*>(defp_->mutable_cpu_data());
    d_defw = this->blobs_[0]->mutable_cpu_diff();
//    memset(d_defw, 0, channels_*4* sizeof(Dtype));
    dh = dh_.mutable_cpu_data();
    dv = dv_.mutable_cpu_data();
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2";
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int ch = 0; ch < channels_; ++ch) {
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.1";
            int vstart = Iv[(n*channels_+ch)*N_+defp[ch]];
            int hstart = Ih[(n*channels_+ch)*N_+defp[ch]];
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.2";
            int vend = min(vstart + kernel_size_, width_);
            int hend = min(hstart + kernel_size_, height_);
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.3";
            Dtype t_dif = top_diff[0];
//            Dtype t_dif = top_diff[n*channels_+ch];
/*			if ( (n==0) && (ch==0) )
				for (int h = hstart; h < hend; ++h) {
					for (int v = vstart; v < vend; ++v) {
					bottom_diff[h * width_ + v] = t_dif;
				  }
				} 
			else*/

/*				for (int h = hstart; h < hend; ++h) {
					for (int v = vstart; v < vend; ++v) {
					bottom_diff[h * width_ + v] = t_dif;
				  }
				}
*/
				bottom_diff[hstart * width_ + vstart] += t_dif;

/*				if (fabs(t_dif) > 1)
					LOG(INFO) << "bp t_dif>1:" << t_dif << ", "<< ch << "," <<n;
				if ( (n==0) && (fabs(t_dif) > 0.01))
					LOG(INFO) << "Backward_gpu!!:" << "," << kernel_size_ <<"," <<width_ <<"," << bottom_diff[hstart * width_ + vstart] << ","<< hstart<<"," <<hend <<","<< vstart <<","<<vend<<","<<(*bottom)[0]->offset(0, 1);
*/

//					LOG(INFO) << "Backward_gpu:" << "," << kernel_size_ <<"," <<width_ <<"," << bottom_diff[hstart * width_ + vstart] << ","<< hstart<<"," <<hend <<","<< vstart <<","<<vend<<","<<(*bottom)[0]->offset(0, 1);
            // dv for 0 ann 1, dh for 2 and 3
   // LOG(INFO) << "bp t_dif:" << t_dif << ", "<< ch << "," << dv[n*channels_+ch]<< ", " << dh[n*channels_+ch]<< "," << top[0]+>offset(0, 1);
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.4";
/*                d_defw[ch*4+0] = 0;
                d_defw[ch*4+1] = 0;
                d_defw[ch*4+2] = 0;
                d_defw[ch*4+3] = 0;*/
            if (n==0)
            {
                d_defw[ch*4+0] =-t_dif*dv[n*channels_+ch]*dv[n*channels_+ch];
                d_defw[ch*4+1] =-t_dif*dv[n*channels_+ch];
                d_defw[ch*4+2] =-t_dif*dh[n*channels_+ch]*dh[n*channels_+ch];
                d_defw[ch*4+3] =-t_dif*dh[n*channels_+ch];
            }
            else
            {
                d_defw[ch*4+0] -= t_dif*dv[n*channels_+ch]*dv[n*channels_+ch];
                d_defw[ch*4+1] -= t_dif*dv[n*channels_+ch];
                d_defw[ch*4+2] -= t_dif*dh[n*channels_+ch]*dh[n*channels_+ch];
                d_defw[ch*4+3] -= t_dif*dh[n*channels_+ch];
            }
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.5";
//if ( (n==top[0]->num()-1) && (ch==0) )
//	LOG(INFO) <<bottom_diff[0] << ", "<< "bp d_defw[" <<ch <<"]:" << d_defw[ch*4+0] << ", "<< d_defw[ch*4+1]<< ", " << d_defw[ch*4+2]<< ", " << d_defw[ch*4+3];

            bottom_diff += bottom[0]->offset(0, 1);
            top_diff += top[0]->offset(0, 1);            
      }
    }
//LOG(INFO) << "bp d_defw:" << d_defw[0*4+0] << ", "<< d_defw[0*4+1]<< ", " << d_defw[0*4+2]<< ", " << d_defw[0*4+3];
    bottom_diff=bottom[0]->mutable_gpu_diff();
//    top[0]->mutable_gpu_diff();
    this->blobs_[0]->mutable_gpu_diff();
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 3";
    break;
  case PoolingParameter_PoolMethod_DEF_ALL:
 //   LOG(INFO)<<"the entry in DEF_ALL backward";
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF";
    bottom_diff = bottom[0]->mutable_cpu_diff();
    top_diff = top[0]->cpu_diff();
    Ih = Ih_.data();
    Iv = Iv_.data();
    defp = defp_.data();
    d_defw = this->blobs_[0]->mutable_cpu_diff();
    memset(d_defw, 0, channels_*4* sizeof(Dtype));
    dh = dh_.mutable_cpu_data();
    dv = dv_.mutable_cpu_data();
    N2 = pooled_width_*pooled_height_;
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2";
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int ch = 0; ch < channels_; ++ch) {
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.1";
            int vstart = Iv[(n*channels_+ch)*N_+defp[ch]];
            int hstart = Ih[(n*channels_+ch)*N_+defp[ch]];
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.2";
            int vend = min(vstart + kernel_size_, width_);
            int hend = min(hstart + kernel_size_, height_);
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.3";
		  //for (int ph = 0; ph < pooled_height_; ++ph) {
			  int ph = static_cast<int>(ch/pooled_width_);
			  int pw = ch - ph*pooled_width_;
			  int phS = ph * stride_;
			//  for (int pw = 0; pw < pooled_width_; ++pw) {
				  int pwS = pw * stride_;
				  int ihb = Ih[(n*channels_+ch)*N_+phS * width_ + pwS];
				  int ivb = Iv[(n*channels_+ch)*N_+phS * width_ + pwS];
				  Dtype t_dif = top_diff[ph * pooled_width_ + pw];
				  bottom_diff[ihb * width_ + ivb] += t_dif;
//				  LOG(INFO) << "bp: " << ph << "*" << pooled_width_ <<"+" << pw << "=>" << ihb << "*" << width_ << "+" << ivb;
			Dtype D1, D2, D3, D4;
				  D1 = dv[(n*channels_+ch)*N2+ph * pooled_width_ + pw]*dv[(n*channels_+ch)*N2+ph * pooled_width_ + pw];
				  D2 = dv[(n*channels_+ch)*N2+ph * pooled_width_ + pw];
				  D3 = dh[(n*channels_+ch)*N2+ph * pooled_width_ + pw]*dh[(n*channels_+ch)*N2+ph * pooled_width_ + pw];
				  D4 = dh[(n*channels_+ch)*N2+ph * pooled_width_ + pw];
				  if (n==0)
				  {
					  d_defw[ch*4+0] =-t_dif*D1;
					  d_defw[ch*4+1] =-t_dif*D2;
					  d_defw[ch*4+2] =-t_dif*D3;
					  d_defw[ch*4+3] =-t_dif*D4;
				  }
				  else
				  {
					  d_defw[ch*4+0] -= t_dif*D1;
					  d_defw[ch*4+1] -= t_dif*D2;
					  d_defw[ch*4+2] -= t_dif*D3;
					  d_defw[ch*4+3] -= t_dif*D4;
				  }
			 // }
		 // }
		//            Dtype t_dif = top_diff[0];
		//			bottom_diff[hstart * width_ + vstart] += t_dif;


//					LOG(INFO) << "Backward_gpu:" << "," << kernel_size_ <<"," <<width_ <<"," << bottom_diff[hstart * width_ + vstart] << ","<< hstart<<"," <<hend <<","<< vstart <<","<<vend<<","<<(*bottom)[0]->offset(0, 1);
            // dv for 0 ann 1, dh for 2 and 3
   // LOG(INFO) << "bp t_dif:" << t_dif << ", "<< ch << "," << dv[n*channels_+ch]<< ", " << dh[n*channels_+ch]<< "," << top[0]+>offset(0, 1);
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.4";
/*                d_defw[ch*4+0] = 0;
                d_defw[ch*4+1] = 0;
                d_defw[ch*4+2] = 0;
                d_defw[ch*4+3] = 0;*/
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.5";
//if ( (n==top[0]->num()-1) && (ch==0) )
//	LOG(INFO) <<bottom_diff[0] << ", "<< "bp d_defw[" <<ch <<"]:" << d_defw[ch*4+0] << ", "<< d_defw[ch*4+1]<< ", " << d_defw[ch*4+2]<< ", " << d_defw[ch*4+3];
//		  LOG(INFO) <<"bp offset:" << (*bottom)[0]->offset(0, 1) << "," <<top[0]->offset(0, 1);
            bottom_diff += bottom[0]->offset(0, 1);
            top_diff += top[0]->offset(0, 1);            
      }
    }
//LOG(INFO) << "bp d_defw:" << d_defw[0*4+0] << ", "<< d_defw[0*4+1]<< ", " << d_defw[0*4+2]<< ", " << d_defw[0*4+3];
    bottom_diff=bottom[0]->mutable_gpu_diff();
//    top[0]->mutable_gpu_diff();
    this->blobs_[0]->mutable_gpu_diff();
 //   LOG(INFO)<<"the end in DEF_ALL backward";
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 3";
    break;
  case PoolingParameter_PoolMethod_DEF_ALL2:
    bottom_diff = bottom[0]->mutable_cpu_diff();
    top_diff = top[0]->cpu_diff();
    Ih = Ih_.data();
    Iv = Iv_.data();
    defp = defp_.data();
    d_defw = this->blobs_[0]->mutable_cpu_diff();
    memset(d_defw, 0, channels_*4* sizeof(Dtype));
    dh = dh_.mutable_cpu_data();
    dv = dv_.mutable_cpu_data();
	N2 = pooled_width_*pooled_height_;
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int ch = 0; ch < channels_; ++ch) {
            int vstart = Iv[(n*channels_+ch)*N_+defp[ch]];
            int hstart = Ih[(n*channels_+ch)*N_+defp[ch]];
            int vend = min(vstart + kernel_size_, width_);
            int hend = min(hstart + kernel_size_, height_);
		  for (int ph = 0; ph < pooled_height_; ++ph) {
			  int phS = ph * stride_;
			  for (int pw = 0; pw < pooled_width_; ++pw) {
				  int pwS = pw * stride_;
				  int ihb = Ih[(n*channels_+ch)*N_+phS * width_ + pwS];
				  int ivb = Iv[(n*channels_+ch)*N_+phS * width_ + pwS];
				  Dtype t_dif = top_diff[ph * pooled_width_ + pw];
				  bottom_diff[ihb * width_ + ivb] += t_dif;
//				  LOG(INFO) << "bp: " << ph << "*" << pooled_width_ <<"+" << pw << "=>" << ihb << "*" << width_ << "+" << ivb;
			Dtype D1, D2, D3, D4;
				  D1 = dv[(n*channels_+ch)*N2+ph * pooled_width_ + pw]*dv[(n*channels_+ch)*N2+ph * pooled_width_ + pw];
				  D2 = dv[(n*channels_+ch)*N2+ph * pooled_width_ + pw];
				  D3 = dh[(n*channels_+ch)*N2+ph * pooled_width_ + pw]*dh[(n*channels_+ch)*N2+ph * pooled_width_ + pw];
				  D4 = dh[(n*channels_+ch)*N2+ph * pooled_width_ + pw];
				  if (n==0)
				  {
					  d_defw[ch*4+0] =-t_dif*D1;
					  d_defw[ch*4+1] =-t_dif*D2;
					  d_defw[ch*4+2] =-t_dif*D3;
					  d_defw[ch*4+3] =-t_dif*D4;
				  }
				  else
				  {
					  d_defw[ch*4+0] -= t_dif*D1;
					  d_defw[ch*4+1] -= t_dif*D2;
					  d_defw[ch*4+2] -= t_dif*D3;
					  d_defw[ch*4+3] -= t_dif*D4;
				  }
			  }
		  }
		//            Dtype t_dif = top_diff[0];
		//			bottom_diff[hstart * width_ + vstart] += t_dif;


//					LOG(INFO) << "Backward_gpu:" << "," << kernel_size_ <<"," <<width_ <<"," << bottom_diff[hstart * width_ + vstart] << ","<< hstart<<"," <<hend <<","<< vstart <<","<<vend<<","<<(*bottom)[0]->offset(0, 1);
            // dv for 0 ann 1, dh for 2 and 3
   // LOG(INFO) << "bp t_dif:" << t_dif << ", "<< ch << "," << dv[n*channels_+ch]<< ", " << dh[n*channels_+ch]<< "," << top[0]+>offset(0, 1);
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.4";
/*                d_defw[ch*4+0] = 0;
                d_defw[ch*4+1] = 0;
                d_defw[ch*4+2] = 0;
                d_defw[ch*4+3] = 0;*/
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.5";
//if ( (n==top[0]->num()-1) && (ch==0) )
//	LOG(INFO) <<bottom_diff[0] << ", "<< "bp d_defw[" <<ch <<"]:" << d_defw[ch*4+0] << ", "<< d_defw[ch*4+1]<< ", " << d_defw[ch*4+2]<< ", " << d_defw[ch*4+3];
//		  LOG(INFO) <<"bp offset:" << (*bottom)[0]->offset(0, 1) << "," <<top[0]->offset(0, 1);
            bottom_diff += bottom[0]->offset(0, 1);
            top_diff += top[0]->offset(0, 1);            
      }
    }
//LOG(INFO) << "bp d_defw:" << d_defw[0*4+0] << ", "<< d_defw[0*4+1]<< ", " << d_defw[0*4+2]<< ", " << d_defw[0*4+3];
    bottom_diff=bottom[0]->mutable_gpu_diff();
//    top[0]->mutable_gpu_diff();
    this->blobs_[0]->mutable_gpu_diff();
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 3";
    break;
  case PoolingParameter_PoolMethod_DEF_ALL3:
  case PoolingParameter_PoolMethod_DEF_ALL4:
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF_ALL3";
	bottom_diff = bottom[0]->mutable_cpu_diff();
    top_diff = top[0]->cpu_diff();
    Ih = Ih_.data();
    Iv = Iv_.data();
    defp = defp_.data();
    d_defw = this->blobs_[0]->mutable_cpu_diff();
    memset(d_defw, 0, channels_*kernel_size_*kernel_size_* sizeof(Dtype));
    dh = dh_.mutable_cpu_data();
    dv = dv_.mutable_cpu_data();
	N2 = pooled_width_*pooled_height_;
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int ch = 0; ch < channels_; ++ch) {
/*            int vstart = Iv[(n*channels_+ch)*N_+defp[ch]];
            int hstart = Ih[(n*channels_+ch)*N_+defp[ch]];
            int vend = min(vstart + kernel_size_, width_);
            int hend = min(hstart + kernel_size_, height_);*/
		  for (int ph = 0; ph < pooled_height_; ++ph) {
			  int phS = ph * stride_;
			  for (int pw = 0; pw < pooled_width_; ++pw) {
				  int pwS = pw * stride_;
				  int idxInv = Ih[(n*channels_+ch)*N2+ph * pooled_width_ + pw];
				  Dtype t_dif = top_diff[ph * pooled_width_ + pw];
				  bottom_diff[idxInv] += t_dif;
				  int defidx = dh[(n*channels_+ch)*N2+ph * pooled_width_ + pw];
//				  if ( (defidx > Nparam) || (defidx < 0) )
//					  LOG(INFO) << "bp: " << idxInv << "," << ph << "*" << pooled_width_ <<"+" << pw << "=>" << idxInv;

//				  LOG(INFO) << "bp: " << ph << "*" << pooled_width_ <<"+" << pw << "=>" << ihb << "*" << width_ << "+" << ivb;
				  d_defw[ch*Nparam+defidx] -= t_dif;
			  }
		  }
		//            Dtype t_dif = top_diff[0];
		//			bottom_diff[hstart * width_ + vstart] += t_dif;


//					LOG(INFO) << "Backward_gpu:" << "," << kernel_size_ <<"," <<width_ <<"," << bottom_diff[hstart * width_ + vstart] << ","<< hstart<<"," <<hend <<","<< vstart <<","<<vend<<","<<(*bottom)[0]->offset(0, 1);
            // dv for 0 ann 1, dh for 2 and 3
   // LOG(INFO) << "bp t_dif:" << t_dif << ", "<< ch << "," << dv[n*channels_+ch]<< ", " << dh[n*channels_+ch]<< "," << top[0]+>offset(0, 1);
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.4";
/*                d_defw[ch*4+0] = 0;
                d_defw[ch*4+1] = 0;
                d_defw[ch*4+2] = 0;
                d_defw[ch*4+3] = 0;*/
//LOG(INFO) << "Backward_gpu PoolingParameter_PoolMethod_DEF 2.5";
//if ( (n==top[0]->num()-1) && (ch==0) )
//	LOG(INFO) <<bottom_diff[0] << ", "<< "bp d_defw[" <<ch <<"]:" << d_defw[ch*4+0] << ", "<< d_defw[ch*4+1]<< ", " << d_defw[ch*4+2]<< ", " << d_defw[ch*4+3];
//		  LOG(INFO) <<"bp offset:" << (*bottom)[0]->offset(0, 1) << "," <<top[0]->offset(0, 1);
            bottom_diff += bottom[0]->offset(0, 1);
            top_diff += top[0]->offset(0, 1);            
      }
    }
//LOG(INFO) << "bp d_defw:" << d_defw[0*4+0] << ", "<< d_defw[0*4+1]<< ", " << d_defw[0*4+2]<< ", " << d_defw[0*4+3];
    bottom_diff=bottom[0]->mutable_gpu_diff();
    this->blobs_[0]->mutable_gpu_diff();
    break;
  case PoolingParameter_PoolMethod_LOWRES:
	  // this Lowres does not require bp
	  break;
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
