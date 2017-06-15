
/**
 * developed by zhujin
 */
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>


#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/proto/caffe.pb.h"



namespace caffe {

template <typename Dtype>
__global__ void kernel_BiLinearResize(const int nthreads, const Dtype* src_data, const int src_height, const int src_width,
		Dtype* dst_data, const int dst_height, const int dst_width, const Dtype scale_h, const Dtype scale_w)
{

	CUDA_KERNEL_LOOP(i, nthreads) {
		int dst_h = i /dst_width;
		Dtype fh = dst_h * scale_h;
		const int src_h = floor(fh);
		fh -= src_h;
		const Dtype w_h0 = std::abs(1.0f - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = i %dst_width;
		Dtype fw = dst_w * scale_w;
		const int src_w = floor(fw);
		fw -= src_w;
		const Dtype w_w0 = std::abs(1.0f - fw);
		const Dtype w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;

		const int src_idx = src_offset_1 + src_w;

		Dtype res = (w_h0 * w_w0 * src_data[src_idx]);

		if (src_w + 1 < src_width)
			res += (w_h0 * w_w1 * src_data[src_idx + 1]);
		if (src_h + 1 < src_height)
			res += (w_h1 * w_w0 * src_data[src_idx + src_width]);

		if (src_w + 1 < src_width && src_h + 1 < src_height)
			res += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);

		dst_data[dst_idx] = res;
	}
}


template <typename Dtype>
void BiLinearResizeMat_gpu(const Dtype* src, const int src_height, const int src_width,
		Dtype* dst, const int dst_height, const int dst_width)
{
	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;


	int loop_n = dst_height * dst_width;
	kernel_BiLinearResize<Dtype> <<<CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >>>(
			loop_n,src, src_height, src_width, dst, dst_height, dst_width, scale_h, scale_w);

	//CUDA_POST_KERNEL_CHECK;
}


template void BiLinearResizeMat_gpu(const float* src, const int src_height, const int src_width,
		float* dst, const int dst_height, const int dst_width);

template void BiLinearResizeMat_gpu(const double* src, const int src_height, const int src_width,
		double* dst, const int dst_height, const int dst_width);



template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src, const int src_n, const int src_c,
		Blob<Dtype>* dst, const int dst_n, const int dst_c) {


	const int src_channels = src->channels();
	const int src_height = src->height();
	const int src_width = src->width();
	const int src_offset = (src_n * src_channels + src_c) * src_height * src_width;

	const int dst_channels = dst->channels();
	const int dst_height = dst->height();
	const int dst_width = dst->width();
	const int dst_offset = (dst_n * dst_channels + dst_c) * dst_height * dst_width;

	const Dtype* src_data = &(src->gpu_data()[src_offset]);
	Dtype* dst_data = &(dst->mutable_gpu_data()[dst_offset]);
	BiLinearResizeMat_gpu(src_data,  src_height,  src_width,
			dst_data,  dst_height,  dst_width);
	CUDA_POST_KERNEL_CHECK;
}

template void ResizeBlob_gpu(const Blob<float>* src, const int src_n, const int src_c,
		Blob<float>* dst, const int dst_n, const int dst_c);
template void ResizeBlob_gpu(const Blob<double>* src, const int src_n, const int src_c,
		Blob<double>* dst, const int dst_n, const int dst_c);

template <typename Dtype>
__global__ void kernel_GetBiLinearResizeMatRules(const int nthreads,  const int src_height, const int src_width,
		const int dst_height, const int dst_width, const Dtype scale_h, const Dtype scale_w,
		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
				Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		int dst_h = index /dst_width;
		Dtype fh = dst_h * scale_h;
		const int src_h = floor(fh);
		fh -= src_h;
		const Dtype w_h0 = std::abs(1.0f - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = index %dst_width;
		Dtype fw = dst_w * scale_w;
		const int src_w = floor(fw);
		fw -= src_w;
		const Dtype w_w0 = std::abs(1.0f - fw);
		const Dtype w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;
//		dst_data[dst_idx] = 0;

		const int src_idx = src_offset_1 + src_w;

		loc1[dst_idx] = src_idx;
		weight1[dst_idx] = w_h0 * w_w0;

		loc2[dst_idx] = 0;
		weight2[dst_idx] = 0;

		weight3[dst_idx] = 0;
		loc3[dst_idx] = 0;

		loc4[dst_idx] = 0;
		weight4[dst_idx] = 0;

		if (src_w + 1 < src_width)
		{
			loc2[dst_idx] = src_idx + 1;
			weight2[dst_idx] = w_h0 * w_w1;
//			dst_data[dst_idx] += (w_h0 * w_w1 * src_data[src_idx + 1]);
		}

		if (src_h + 1 < src_height)
		{
//			dst_data[dst_idx] += (w_h1 * w_w0 * src_data[src_idx + src_width]);
			weight3[dst_idx] = w_h1 * w_w0;
			loc3[dst_idx] = src_idx + src_width;
		}

		if (src_w + 1 < src_width && src_h + 1 < src_height)
		{
			loc4[dst_idx] = src_idx + src_width + 1;
			weight4[dst_idx] = w_h1 * w_w1;
//			dst_data[dst_idx] += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
		}

	}
}



template <typename Dtype>
__global__ void kernel_ResizeBlob(const int nthreads,const int num,const int channels, const Dtype* src, const int src_height, const int src_width,
		Dtype* dst, const int dst_height, const int dst_width, const Dtype scale_h, const Dtype scale_w)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		int i = index %( dst_height * dst_width);
		int c = (index/(dst_height * dst_width))%channels;
		int n = (index/(dst_height * dst_width))/channels;
		int src_offset = (n * channels + c) * src_height * src_width;
		int dst_offset = (n * channels + c) * dst_height * dst_width;

		const Dtype* src_data = src+src_offset;
		Dtype* dst_data = dst+dst_offset;

		int dst_h = i /dst_width;
		Dtype fh = dst_h * scale_h;
		const int src_h = floor(fh);
		fh -= src_h;
		const Dtype w_h0 = std::abs(1.0f - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = i %dst_width;
		Dtype fw = dst_w * scale_w;
		const int src_w = floor(fw);
		fw -= src_w;
		const Dtype w_w0 = std::abs(1.0f - fw);
		const Dtype w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;


		const int src_idx = src_offset_1 + src_w;
		Dtype res = (w_h0 * w_w0 * src_data[src_idx]);

		if (src_w + 1 < src_width)
			res += (w_h0 * w_w1 * src_data[src_idx + 1]);
		if (src_h + 1 < src_height)
			res += (w_h1 * w_w0 * src_data[src_idx + src_width]);

		if (src_w + 1 < src_width && src_h + 1 < src_height)
			res += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);

		dst_data[dst_idx] = res;
	}
}

/*
// new version by Sifei Liu
template <typename Dtype>
__global__ void kernel_ResizeBlob(const int nthreads,const int num,const int channels, const Dtype* src, const int src_height, const int src_width,
        Dtype* dst, const int dst_height, const int dst_width, const Dtype scale_h, const Dtype scale_w)
{
    CUDA_KERNEL_LOOP(index, nthreads) {
        int d_i = index %( dst_height * dst_width);
        int d_c = (index/(dst_height * dst_width))%channels;
        int d_n = (index/(dst_height * dst_width))/channels;

        int s_c = (index/(src_height * src_width))%channels;
        int s_n = (index/(src_height * src_width))/channels;

        int src_offset = (s_n * channels + s_c) * src_height * src_width;
        int dst_offset = (d_n * channels + d_c) * dst_height * dst_width;

        const Dtype* src_data = src+src_offset;
        Dtype* dst_data = dst+dst_offset;

        int dst_h = d_i /dst_width;
        Dtype fh = dst_h * scale_h;
        const int src_h = floor(fh);
        fh -= src_h;
        const Dtype w_h0 = std::abs(1.0f - fh);
        const Dtype w_h1 = std::abs(fh);

        const int dst_offset_1 =  dst_h * dst_width;
        const int src_offset_1 =  src_h * src_width;

        int dst_w = d_i %dst_width;
        Dtype fw = dst_w * scale_w;
        const int src_w = floor(fw);
        fw -= src_w;
        const Dtype w_w0 = std::abs(1.0f - fw);
        const Dtype w_w1 = std::abs(fw);

        const int dst_idx = dst_offset_1 + dst_w;

        const int src_idx = src_offset_1 + src_w;
        Dtype res = (w_h0 * w_w0 * src_data[src_idx]);

        if (src_w + 1 < src_width)
            res += (w_h0 * w_w1 * src_data[src_idx + 1]);
        if (src_h + 1 < src_height)
            res += (w_h1 * w_w0 * src_data[src_idx + src_width]);

        if (src_w + 1 < src_width && src_h + 1 < src_height)
            res += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);

        dst_data[dst_idx] = res;
    }
}*/


template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src,Blob<Dtype>* dst) {

	CHECK(src->num() == dst->num())<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())<< "src->channels() == dst->channels()";

	const int src_num = src->num();
	const int src_channels = src->channels();
	const int src_height = src->height();
	const int src_width = src->width();


	const int dst_channels = dst->channels();
	const int dst_height = dst->height();
	const int dst_width = dst->width();


	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;
	int loop_n = dst_height * dst_width*dst_channels*src_num;
	const Dtype* src_data = src->gpu_data();
	Dtype* dst_data = dst->mutable_gpu_data();
	kernel_ResizeBlob<Dtype> <<<CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >>>(loop_n,src_num,src_channels,
			src_data, src_height,src_width,
			dst_data, dst_height, dst_width,
			scale_h,scale_w);
	CUDA_POST_KERNEL_CHECK;
}



template void ResizeBlob_gpu(const Blob<float>* src,
		Blob<float>* dst);
template void ResizeBlob_gpu(const Blob<double>* src,
		Blob<double>* dst);


template <typename Dtype>
void GetBiLinearResizeMatRules_gpu( const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4)
{
	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;


	int loop_n = dst_height * dst_width;

	kernel_GetBiLinearResizeMatRules<Dtype> <<<CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >>>(
			loop_n,  src_height,  src_width,
			dst_height, dst_width, scale_h, scale_w,
			loc1,  weight1,  loc2,  weight2,
			loc3,  weight3,   loc4,   weight4);
	CUDA_POST_KERNEL_CHECK;
}

template void GetBiLinearResizeMatRules_gpu(  const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		float* loc1, float* weight1, float* loc2, float* weight2,
				float* loc3, float* weight3, float* loc4, float* weight4);

template void GetBiLinearResizeMatRules_gpu(  const int src_height, const int src_width,
		 const int dst_height, const int dst_width,
		double* loc1, double* weight1, double* loc2, double* weight2,
				double* loc3, double* weight3, double* loc4, double* weight4);



template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src,Blob<Dtype>* dst,
		Blob<Dtype>* loc1, Blob<Dtype>* loc2, Blob<Dtype>* loc3, Blob<Dtype>* loc4){

	CHECK(src->num() == dst->num())<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())<< "src->channels() == dst->channels()";

	GetBiLinearResizeMatRules_gpu(  src->height(),src->width(),
			 dst->height(), dst->width(),
			loc1->mutable_gpu_data(), loc1->mutable_gpu_diff(), loc2->mutable_gpu_data(), loc2->mutable_gpu_diff(),
			loc3->mutable_gpu_data(), loc3->mutable_gpu_diff(), loc4->mutable_gpu_data(), loc4->mutable_gpu_diff());

	ResizeBlob_gpu( src,  dst) ;

}
template void ResizeBlob_gpu(const Blob<float>* src,Blob<float>* dst,
		Blob<float>* loc1, Blob<float>* loc2, Blob<float>* loc3, Blob<float>* loc4);
template void ResizeBlob_gpu(const Blob<double>* src,Blob<double>* dst,
		Blob<double>* loc1, Blob<double>* loc2, Blob<double>* loc3, Blob<double>* loc4);

/*
template <typename Dtype>
void GenerateSubBlobs_gpu(const Blob<Dtype>& src,
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
	caffe::caffe_gpu_set(dst.count(),Dtype(0),dst.mutable_gpu_data());
	for(int n = 0; n < nums_; n++){

		const Dtype*  src_data = src.gpu_data() + src.offset(n);
		Dtype*  dst_data = dst.mutable_gpu_data() + dst.offset(n*height_col_*width_col_);
		caffe::im2col_v2_gpu(src_data, channels_, height_,
	            width_, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
	            dst_data);

	}
}

template void GenerateSubBlobs_gpu(const Blob<float>& src,
		Blob<float>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w);
template void GenerateSubBlobs_gpu(const Blob<double>& src,
		Blob<double>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w);
*/
template <typename Dtype>
__global__ void kernel_CropBlob(const int nthreads, const Dtype* src_data, Dtype* dst_data,
		const int num, const int channels, const int in_h, const int in_w,
		const int out_h, const int out_w, const int start_h, const int start_w)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		int  n  = index/channels/out_h/out_w;
		int c = (index/(out_h*out_w))% channels;
		int  h = (index%(out_h*out_w))/out_w;
		int  w = (index%(out_h*out_w))%out_w;

		Dtype* dst_data_ptr =dst_data+ ((n* channels+c)*out_h  )*out_w  ;

		const Dtype* src_data_ptr = src_data +  ((n* channels+c)*in_h  )*in_w  ;
		dst_data_ptr[h*out_w+w] = src_data_ptr[(h+start_h)*in_w + w+start_w];

	}
}


template <typename Dtype>
void CropBlobs_gpu( const Blob<Dtype>&src,
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
	CHECK(out_h <=in_h) <<" out_h should nor be larger than input_height";
	CHECK(out_w <=in_w) <<" out_w should nor be larger than input_width";

	dst.Reshape(num,channels,out_h,out_w);

	if((out_h != in_h) || (out_w != in_w)){
		const int loop_n = num*channels*out_h*out_w;

		kernel_CropBlob <Dtype> <<< CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >>> (loop_n,
				src.gpu_data(),  dst.mutable_gpu_data(),
				 num,  channels,  in_h,  in_w, out_h,  out_w,   start_h,   start_w);

	}
	else
	{
		caffe::caffe_copy(src.count(),src.gpu_data(),dst.mutable_gpu_data());
	}
}

template void  CropBlobs_gpu( const Blob<float>&src,
		const int start_h, const int start_w,
				const int end_h, const int end_w, Blob<float>&dst);

template void  CropBlobs_gpu( const Blob<double>&src,
		const int start_h, const int start_w,
				const int end_h, const int end_w, Blob<double>&dst);


template <typename Dtype>
__global__ void kernel_CropBlob(const int nthreads, const Dtype* src_data, Dtype* dst_data,
		const int num, const int channels, const int in_h, const int in_w,
		const int dst_num, const int dst_h, const int dst_w,
		const int src_num_id, const int dst_num_id,const int out_h, const int out_w,
		const int start_h, const int start_w, const int dst_start_h, const int dst_start_w){
	CUDA_KERNEL_LOOP(index, nthreads) {
		int c = (index/(out_h*out_w))% channels;
		int  h = (index%(out_h*out_w))/out_w;
		int  w = (index%(out_h*out_w))%out_w;

		Dtype* dst_data_ptr =dst_data+ ((dst_num_id* channels+c)*dst_h  )*dst_w  ;

		const Dtype* src_data_ptr = src_data +  ((src_num_id* channels+c)*in_h  )*in_w  ;
		int true_src_h = h+start_h;
		int true_dst_h = h+dst_start_h;
		int true_src_w = w+start_w;
		int true_dst_w = w + dst_start_w;
		if(true_src_h >= 0 && true_src_h < in_h && true_src_w >= 0 && true_src_w < in_w &&
				true_dst_h >= 0 && true_dst_h < dst_h && true_dst_w>= 0 && true_dst_w< dst_w	)
			dst_data_ptr[true_dst_h *dst_w + true_dst_w] =
				src_data_ptr[true_src_h * in_w + true_src_w];
	}
}


template <typename Dtype>
void CropBlobs_gpu( const Blob<Dtype>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst,
		const int dst_num_id,const int dst_start_h  , const int dst_start_w ){

	const int in_h = src.height();
	const int in_w = src.width();
	const int dst_h = dst.height();
	const int dst_w = dst.width();
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

	const int loop_n = channels*out_h*out_w;

	kernel_CropBlob <Dtype> <<< CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >>> (loop_n,
					src.gpu_data(),  dst.mutable_gpu_data(),
					src.num(),  channels,  in_h,  in_w,
					dst.num(),dst_h,dst_w, src_num_id,dst_num_id,
					out_h,  out_w,   start_h,   start_w, dst_start_h, dst_start_w);

}

template void CropBlobs_gpu( const Blob<float>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<float>&dst,
		const int dst_num_id,const int dst_start_h  , const int dst_start_w  );

template void CropBlobs_gpu( const Blob<double>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<double>&dst,
		const int dst_num_id,const int dst_start_h , const int dst_start_w );



/*
template <typename Dtype>
void ConcateSubImagesInBlobs_gpu(const Blob<Dtype>& src,
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
			const Dtype*  src_data = src.gpu_data() + src.offset(n*height_col_*width_col_);
			Dtype*  dst_data = dst.mutable_gpu_data() + dst.offset(n);
			caffe::col2im_v2_gpu(src_data, channels_, height_,
		            width_, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
		            dst_data);
	}

	return;
}

template void ConcateSubImagesInBlobs_gpu(const Blob<float>& src,
		Blob<float>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w, const int out_img_h, const int out_img_w);

template void ConcateSubImagesInBlobs_gpu(const Blob<double>& src,
		Blob<double>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w, const int out_img_h, const int out_img_w);
*/



// namespace caffe
}
