
/**
 * developed by alan and zhujin
 */
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>


#include <cmath>
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"


/**
 * @todo parallelization for all image processing
 */
namespace caffe {

template <typename Dtype>
void BiLinearResizeMat_cpu(const Dtype* src, const int src_height, const int src_width,
		Dtype* dst, const int dst_height, const int dst_width)
{
	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;
	Dtype* dst_data = dst;
	const Dtype* src_data = src;

	for(int dst_h = 0; dst_h < dst_height; ++dst_h){
		Dtype fh = dst_h * scale_h;

		int src_h = std::floor(fh);

		fh -= src_h;
		const Dtype w_h0 = std::abs((Dtype)1.0 - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		Dtype* dst_data_ptr = dst_data + dst_offset_1;

		for(int dst_w = 0 ; dst_w < dst_width; ++dst_w){

			Dtype fw = dst_w * scale_w;
			int src_w = std::floor(fw);
			fw -= src_w;
			const Dtype w_w0 = std::abs((Dtype)1.0 - fw);
			const Dtype w_w1 = std::abs(fw);


			Dtype dst_value = 0;

			const int src_idx = src_offset_1 + src_w;
			dst_value += (w_h0 * w_w0 * src_data[src_idx]);
			int flag = 0;
			if (src_w + 1 < src_width){
				dst_value += (w_h0 * w_w1 * src_data[src_idx + 1]);
				++flag;
			}
			if (src_h + 1 < src_height){
				dst_value += (w_h1 * w_w0 * src_data[src_idx + src_width]);
				++flag;
			}

			if (flag>1){
				dst_value += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
//				++flag;
			}
			*(dst_data_ptr++) = dst_value;
		}
	}

}


template void BiLinearResizeMat_cpu(const float* src, const int src_height, const int src_width,
		float* dst, const int dst_height, const int dst_width);

template void BiLinearResizeMat_cpu(const double* src, const int src_height, const int src_width,
		double* dst, const int dst_height, const int dst_width);

template <typename Dtype>
void RuleBiLinearResizeMat_cpu(const Dtype* src,
		Dtype* dst, const int dst_h, const int dst_w,
		const Dtype* loc1, const Dtype* weight1, const Dtype* loc2,const Dtype* weight2,
		const	Dtype* loc3,const Dtype* weight3,const Dtype* loc4, const Dtype* weight4)
{

	Dtype* dst_data = dst;
	const Dtype* src_data = src;

	int loop_n = dst_h  * dst_w ;
	for(int i=0 ; i< loop_n; i++)
	{


		dst_data[i] += (weight1[i] * src_data[static_cast<int>(loc1[i])]);
		dst_data[i] += (weight2[i] * src_data[static_cast<int>(loc2[i])]);
		dst_data[i] += (weight3[i] * src_data[static_cast<int>(loc3[i])]);
		dst_data[i] += (weight4[i] * src_data[static_cast<int>(loc4[i])]);

	}

}

template void RuleBiLinearResizeMat_cpu(const float* src,
		float* dst, const int dst_h, const int dst_w,
		const float* loc1, const float* weight1, const float* loc2,const float* weight2,
		const	float* loc3,const float* weight3,const float* loc4, const float* weight4);
template void RuleBiLinearResizeMat_cpu(const double* src,
		double* dst, const int dst_h, const int dst_w,
		const double* loc1, const double* weight1, const double* loc2,const double* weight2,
		const	double* loc3,const double* weight3,const double* loc4, const double* weight4);



template <typename Dtype>
void GetBiLinearResizeMatRules_cpu( const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4)
{
	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;


	int loop_n = dst_height * dst_width;


	for(int i=0 ; i< loop_n; i++)
	{
		int dst_h = i /dst_width;
		Dtype fh = dst_h * scale_h;
		int src_h ;
		if(typeid(Dtype).name() == typeid(double).name())
			 src_h = floor(fh);
		else
			 src_h = floorf(fh);

		fh -= src_h;
		const Dtype w_h0 = std::abs((Dtype)1.0 - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = i %dst_width;
		Dtype fw = dst_w * scale_w;

		int src_w ;
		if(typeid(Dtype).name() == typeid(double).name())
			src_w = floor(fw);
		else
			src_w = floorf(fw);

		fw -= src_w;
		const Dtype w_w0 = std::abs((Dtype)1.0 - fw);
		const Dtype w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;
//		dst_data[dst_idx] = 0;

		const int src_idx = src_offset_1 + src_w;

		loc1[dst_idx] = static_cast<Dtype>(src_idx);
		weight1[dst_idx] = w_h0 * w_w0;


		loc2[dst_idx] = 0;
		weight2[dst_idx] = 0;

		weight3[dst_idx] = 0;
		loc3[dst_idx] = 0;

		loc4[dst_idx] = 0;
		weight4[dst_idx] = 0;

		if (src_w + 1 < src_width)
		{
			loc2[dst_idx] = static_cast<Dtype>(src_idx + 1);
			weight2[dst_idx] = w_h0 * w_w1;
//			dst_data[dst_idx] += (w_h0 * w_w1 * src_data[src_idx + 1]);
		}

		if (src_h + 1 < src_height)
		{
//			dst_data[dst_idx] += (w_h1 * w_w0 * src_data[src_idx + src_width]);
			weight3[dst_idx] = w_h1 * w_w0;
			loc3[dst_idx] = static_cast<Dtype>(src_idx + src_width);
		}

		if (src_w + 1 < src_width && src_h + 1 < src_height)
		{
			loc4[dst_idx] = static_cast<Dtype>(src_idx + src_width + 1);
			weight4[dst_idx] = w_h1 * w_w1;
//			dst_data[dst_idx] += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
		}

	}

}


template void GetBiLinearResizeMatRules_cpu(  const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		float* loc1, float* weight1, float* loc2, float* weight2,
				float* loc3, float* weight3, float* loc4, float* weight4);

template void GetBiLinearResizeMatRules_cpu(  const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		double* loc1, double* weight1, double* loc2, double* weight2,
				double* loc3, double* weight3, double* loc4, double* weight4);




template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src, const int src_n, const int src_c,
		Blob<Dtype>* dst, const int dst_n, const int dst_c) {


	const int src_channels = src->channels();
	const int src_height = src->height();
	const int src_width = src->width();
	const int src_offset = (src_n * src_channels + src_c) * src_height * src_width;

	const int dst_channels = dst->channels();
	const int dst_height = dst->height();
	const int dst_width = dst->width();
	const int dst_offset = (dst_n * dst_channels + dst_c) * dst_height * dst_width;


	const Dtype* src_data = &(src->cpu_data()[src_offset]);
	Dtype* dst_data = &(dst->mutable_cpu_data()[dst_offset]);
	BiLinearResizeMat_cpu(src_data,  src_height,  src_width,
			dst_data,  dst_height,  dst_width);
}

template void ResizeBlob_cpu(const Blob<float>* src, const int src_n, const int src_c,
		Blob<float>* dst, const int dst_n, const int dst_c);
template void ResizeBlob_cpu(const Blob<double>* src, const int src_n, const int src_c,
		Blob<double>* dst, const int dst_n, const int dst_c);


template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,Blob<Dtype>* dst)
{
	CHECK(src->num() == dst->num())<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())<< "src->channels() == dst->channels()";

	for(int n=0;n< src->num();++n)
	{
		for(int c=0; c < src->channels() ; ++c)
		{
			ResizeBlob_cpu(src,n,c,dst,n,c);
		}
	}
}
template void ResizeBlob_cpu(const Blob<float>* src,Blob<float>* dst);
template void ResizeBlob_cpu(const Blob<double>* src,Blob<double>* dst);



template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,Blob<Dtype>* dst,
		Blob<Dtype>* loc1, Blob<Dtype>* loc2, Blob<Dtype>* loc3, Blob<Dtype>* loc4){

	CHECK(src->num() == dst->num())<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())<< "src->channels() == dst->channels()";

	GetBiLinearResizeMatRules_cpu(  src->height(),src->width(),
			 dst->height(), dst->width(),
			loc1->mutable_cpu_data(), loc1->mutable_cpu_diff(), loc2->mutable_cpu_data(), loc2->mutable_cpu_diff(),
			loc3->mutable_cpu_data(), loc3->mutable_cpu_diff(), loc4->mutable_cpu_data(), loc4->mutable_cpu_diff());


	ResizeBlob_cpu(src, dst );

}
template void ResizeBlob_cpu(const Blob<float>* src,Blob<float>* dst,
		Blob<float>* loc1, Blob<float>* loc2, Blob<float>* loc3, Blob<float>* loc4);
template void ResizeBlob_cpu(const Blob<double>* src,Blob<double>* dst,
		Blob<double>* loc1, Blob<double>* loc2, Blob<double>* loc3, Blob<double>* loc4);


/**
 *  src.Shape(nums_, channels, height, width)
 *  dst.Reshape(nums_*width_out_*height_out_，channels_, kernel_h, kernel_w);
 */
/*template <typename Dtype>
void GenerateSubBlobs_cpu(const Blob<Dtype>& src,
		Blob<Dtype>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w)
{
	const int nums_ = src.num();
	const int channels_ = src.channels();
	const int height_ = src.height();
	const int width_ = src.width();
	const int height_col_ =(height_ + 2 * pad_h - kernel_h) / stride_h + 1;
	const int width_col_ = (width_ + 2 * pad_w - kernel_w) / stride_w + 1;

	*
	 * actually after im2col_v2, data is stored as
	 * col_buffer_.Reshape(1*height_out_*width_out_, channels_  , kernel_h_ , kernel_w_);
	 * *
	dst.Reshape(height_col_*width_col_*nums_,channels_,  kernel_h, kernel_w);
	caffe_set(dst.count(),Dtype(0),dst.mutable_cpu_data());
	for(int n = 0; n < nums_; n++){

		const Dtype*  src_data = src.cpu_data() + src.offset(n);
		Dtype*  dst_data = dst.mutable_cpu_data() + dst.offset(n*height_col_*width_col_);
		caffe::im2col_v2_cpu(src_data, channels_, height_,
	            width_, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
	            dst_data);

	}
}*/
/*
template void GenerateSubBlobs_cpu(const Blob<float>& src,
		Blob<float>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w);
template void GenerateSubBlobs_cpu(const Blob<double>& src,
		Blob<double>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w);
*/
/**
 *  end_h is not included
 */
template <typename Dtype>
void CropBlobs_cpu( const Blob<Dtype>&src,
		const int start_h, const int start_w,
		const int end_h, const int end_w, Blob<Dtype>&dst)
{
	const int in_h = src.height();
	const int in_w = src.width();
	const int num = src.num();
	const int channels = src.channels();
	const int out_h = end_h - start_h;
	const int out_w = end_w - start_w;
	CHECK(out_h > 0) <<" end_h should be larger than start_h";
	CHECK(out_w > 0) <<" end_w should be larger than start_w";
	CHECK_LE(out_h ,in_h) <<" out_h should nor be larger than input_height";
	CHECK_LE(out_w ,in_w) <<" out_w should nor be larger than input_width";

	dst.Reshape(num,channels,out_h,out_w);
	if((out_h != in_h) || (out_w != in_w)){
		for(int n=0; n < num; n++)
		{
			for(int c=0; c<channels; c++)
			{
				Dtype* dst_data =dst.mutable_cpu_data() + dst.offset(n,c);
				const Dtype* src_data = src.cpu_data() + src.offset(n,c);

				for(int h=0; h< out_h; ++h)
				{
					const Dtype* src_data_p = src_data + (h+start_h)*in_w + start_w;
					Dtype* dst_data_p = dst_data+ h*out_w;
					for(int w=0; w<out_w;++w)
					{
						*(dst_data_p++)= *(src_data_p + w);
					}
				}
			}
		}
	}
	else
	{
		caffe::caffe_copy(src.count(),src.cpu_data(),dst.mutable_cpu_data());
	}
}

template void  CropBlobs_cpu( const Blob<float>&src,
		const int start_h, const int start_w,
				const int end_h, const int end_w, Blob<float>&dst);

template void  CropBlobs_cpu( const Blob<double>&src,
		const int start_h, const int start_w,
				const int end_h, const int end_w, Blob<double>&dst);



template <typename Dtype>
void CropBlobs_cpu( const Blob<Dtype>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst,
		const int dst_num_id,const int dst_start_h  , const int dst_start_w  ){
	const int in_h = src.height();
	const int in_w = src.width();
	const int dst_w = dst.width();
	const int dst_h = dst.height();
	const int channels = src.channels();
	const int out_h = end_h - start_h;
	const int out_w = end_w - start_w;
	CHECK(out_h > 0) <<" end_h should be larger than start_h";
	CHECK(out_w > 0) <<" end_w should be larger than start_w";
//	CHECK(out_h <=in_h) <<" out_h should nor be larger than input_height";
//	CHECK(out_w <=in_w) <<" out_w should nor be larger than input_width";

	CHECK_GT(src.num(), src_num_id);
	CHECK_GT(dst.num(), dst_num_id);
	CHECK_EQ(channels, dst.channels());
//	CHECK_GE(dst.height(), end_h);
//	CHECK_GE(dst.width(), end_w);

	for(int c=0; c<channels; c++)
	{
		Dtype* dst_data =dst.mutable_cpu_data() + dst.offset(dst_num_id,c);
		const Dtype* src_data = src.cpu_data() + src.offset(src_num_id,c);


		for(int h=0; h< out_h; ++h)
		{
			int true_dst_h = h+dst_start_h;
			int true_src_h = h+start_h;
			if(true_dst_h >= 0 && true_dst_h < dst_h && true_src_h >= 0 && true_src_h < in_h)
			{
				int h_off_src = true_src_h*in_w;
				int h_off_dst = true_dst_h*dst_w;

				int true_dst_w =  dst_start_w;
				int true_src_w =  start_w;
				for(int w=0; w<out_w;++w)
				{
					if(true_dst_w >= 0 && true_dst_w < dst_w && true_src_w >= 0 && true_src_w < in_w)
						dst_data[h_off_dst + true_dst_w] = src_data[h_off_src+ true_src_w];
					++true_dst_w;
					++true_src_w;
				}
			}
		}
	}

}

template void CropBlobs_cpu( const Blob<float>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<float>&dst,
		const int dst_num_id,const int dst_start_h , const int dst_start_w );

template void CropBlobs_cpu( const Blob<double>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<double>&dst,
		const int dst_num_id,const int dst_start_h  , const int dst_start_w );

/**
 * src(n,c,h,w)  ===>   dst(n_ori,c,new_h,new_w)
 *
 */
/*template <typename Dtype>
void ConcateSubImagesInBlobs_cpu(const Blob<Dtype>& src,
		Blob<Dtype>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w, const int out_img_h, const int out_img_w)
{
	const int in_nums = src.num();


	const int height_col_ =(out_img_h + 2 * pad_h - kernel_h) / stride_h + 1;
	const int width_col_ = (out_img_w + 2 * pad_w - kernel_w) / stride_w + 1;

//	std::cout<<"in_nums:"<<in_nums<<" kernel_h:"<<kernel_h<<" kernel_w:"<<kernel_w
//			<<" pad_h:"<<pad_h<<" pad_w:"<<pad_w<<" stride_h:"<<stride_h<<
//			" stride_w:"<<stride_w<<"  out_img_h:"<<out_img_h <<" out_img_w:"<<out_img_w
//			<< " height_col:"<<height_col_<<" width_col:"<<width_col_<<std::endl;

	dst.Reshape(in_nums/height_col_/width_col_,src.channels(),  out_img_h, out_img_w);
//	std::cout<<"in_nums/height_col_/width_col_,src.channels(),  out_img_h, out_img_w: "<<
//			in_nums/height_col_/width_col_<< " "<<src.channels()<<"  "<<out_img_h<<"  "<<
//			out_img_w<<std::endl;
	const int channels_ = dst.channels();
	const int height_ = dst.height();
	const int width_ = dst.width();
	const int out_num = dst.num();

	for(int n = 0; n < out_num; n++){
			const Dtype*  src_data = src.cpu_data() + src.offset(n*height_col_*width_col_);
			Dtype*  dst_data = dst.mutable_cpu_data() + dst.offset(n);
			caffe::col2im_v2_cpu(src_data, channels_, height_,
		            width_, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
		            dst_data);
	}

	return;
}*/
/*
template void ConcateSubImagesInBlobs_cpu(const Blob<float>& src,
		Blob<float>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w, const int out_img_h, const int out_img_w);

template void ConcateSubImagesInBlobs_cpu(const Blob<double>& src,
		Blob<double>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w, const int out_img_h, const int out_img_w);
*/


template <typename Dtype>
void CropBlobs( const Blob<Dtype>&src, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst)
{
  switch (Caffe::mode()) {
	case Caffe::CPU:
	  CropBlobs_cpu(src,start_h, start_w,end_h, end_w,dst);
	  break;
#ifndef CPU_ONLY
	case Caffe::GPU:
		CropBlobs_gpu(src,start_h, start_w,end_h, end_w,dst);
	  break;
#endif
	default:
	  LOG(FATAL)<< "Unknown caffe mode.";
  }
}

template void  CropBlobs( const Blob<float>&src,
	const int start_h, const int start_w,
			const int end_h, const int end_w, Blob<float>&dst);

template void  CropBlobs( const Blob<double>&src,
	const int start_h, const int start_w,
			const int end_h, const int end_w, Blob<double>&dst);


template <typename Dtype>
void CropBlobs( const Blob<Dtype>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst,
		const int dst_num_id,const int dst_start_h  , const int dst_start_w ){
  switch (Caffe::mode()) {
	case Caffe::CPU:
	  CropBlobs_cpu( src, src_num_id,  start_h,
				 start_w,  end_h, end_w, dst,
				 dst_num_id,dst_start_h  , dst_start_w );
	  break;
#ifndef CPU_ONLY
	case Caffe::GPU:
	  CropBlobs_gpu( src, src_num_id,  start_h,
					 start_w,  end_h, end_w, dst,
					 dst_num_id, dst_start_h  , dst_start_w );
	  break;
#endif
	default:
	  LOG(FATAL)<< "Unknown caffe mode.";
  }
}

template void CropBlobs( const Blob<float>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<float>&dst,
		const int dst_num_id,const int dst_start_h  , const int dst_start_w );
template void CropBlobs( const Blob<double>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<double>&dst,
		const int dst_num_id,const int dst_start_h  , const int dst_start_w );

template <typename Dtype>
void ResizeBlob(const Blob<Dtype>* src,
		Blob<Dtype>* dst)
{
	switch (Caffe::mode()) {
		case Caffe::CPU:
		  ResizeBlob_cpu(src,dst);
		  break;
#ifndef CPU_ONLY
		case Caffe::GPU:
		  ResizeBlob_gpu(src,dst);
		  break;
#endif
		default:
		  LOG(FATAL)<< "Unknown caffe mode.";
	}
}
template void ResizeBlob(const Blob<float>* src,Blob<float>* dst);
template void ResizeBlob(const Blob<double>* src,Blob<double>* dst);

}// namespace caffe
