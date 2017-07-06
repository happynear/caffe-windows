#ifndef CAFFE_UTIL_UTIL_IMG_H_
#define CAFFE_UTIL_UTIL_IMG_H_
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src, const int src_n, const int src_c,
		Blob<Dtype>* dst, const int dst_n, const int dst_c);

template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,
		Blob<Dtype>* dst);

template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src,Blob<Dtype>* dst,
		Blob<Dtype>* loc1, Blob<Dtype>* loc2, Blob<Dtype>* loc3, Blob<Dtype>* loc4);

#ifndef CPU_ONLY
template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src,Blob<Dtype>* dst,
		Blob<Dtype>* loc1, Blob<Dtype>* loc2, Blob<Dtype>* loc3, Blob<Dtype>* loc4);

template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src,
		Blob<Dtype>* dst);

template <typename Dtype>
void ResizeBlob_gpu(const Blob<Dtype>* src, const int src_n, const int src_c,
		Blob<Dtype>* dst, const int dst_n, const int dst_c);
#endif

template <typename Dtype>
void ResizeBlob(const Blob<Dtype>* src,
		Blob<Dtype>* dst);

template <typename Dtype>
void BiLinearResizeMat_cpu(const Dtype* src, const int src_h, const int src_w,
		Dtype* dst, const int dst_h, const int dst_w);

template <typename Dtype>
void RuleBiLinearResizeMat_cpu(const Dtype* src,Dtype* dst, const int dst_h, const int dst_w,
		const Dtype* loc1, const Dtype* weight1, const Dtype* loc2,const Dtype* weight2,
		const	Dtype* loc3,const Dtype* weight3,const Dtype* loc4, const Dtype* weight4);


template <typename Dtype>
void GetBiLinearResizeMatRules_cpu(  const int src_h, const int src_w,
		  const int dst_h, const int dst_w,
		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4);

#ifndef CPU_ONLY
template <typename Dtype>
void GetBiLinearResizeMatRules_gpu(  const int src_h, const int src_w,
		  const int dst_h, const int dst_w,
		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4);


/**
 * be careful, this function is valid only when src and dst are GPU memory
 *
 */
template <typename Dtype>
void BiLinearResizeMat_gpu(const Dtype* src, const int src_h, const int src_w,
		Dtype* dst, const int dst_h, const int dst_w);
#endif

//
//template <typename Dtype>
//void BlobAddPadding_cpu(const Blob<Dtype>* src,
//		Blob<Dtype>* dst, const int pad_h,const int pad_w);

template <typename Dtype>
void GenerateSubBlobs_cpu(const Blob<Dtype>& src,
		Blob<Dtype>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w);

#ifndef CPU_ONLY
template <typename Dtype>
void GenerateSubBlobs_gpu(const Blob<Dtype>& src,
		Blob<Dtype>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w);

#endif

template <typename Dtype>
void CropBlobs_cpu( const Blob<Dtype>&src, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst);

#ifndef CPU_ONLY
template <typename Dtype>
void CropBlobs_gpu( const Blob<Dtype>&src, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst);
#endif

/**
 * @brief  crop blob. The destination blob will be reshaped.
 */
template <typename Dtype>
void CropBlobs( const Blob<Dtype>&src, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst);



template <typename Dtype>
void CropBlobs_cpu( const Blob<Dtype>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst,
		const int dst_num_id,const int dst_start_h = 0, const int dst_start_w = 0);

#ifndef CPU_ONLY
template <typename Dtype>
void CropBlobs_gpu( const Blob<Dtype>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst,
		const int dst_num_id,const int dst_start_h = 0, const int dst_start_w = 0);
#endif
/**
 * @brief  crop blob. The destination blob will not be reshaped.
 */
template <typename Dtype>
void CropBlobs( const Blob<Dtype>&src, const int src_num_id, const int start_h,
		const int start_w, const int end_h, const int end_w, Blob<Dtype>&dst,
		const int dst_num_id,const int dst_start_h = 0, const int dst_start_w = 0);

/**
 * src(n,c,h,w)  ===>   dst(n_ori,c,new_h,new_w)
 * n contains sub images from (0,0),(0,1),....(nh,nw)
 */
template <typename Dtype>
void ConcateSubImagesInBlobs_cpu( const Blob<Dtype>& src,
		Blob<Dtype>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w, const int out_img_h, const int out_img_w);

#ifndef CPU_ONLY
template <typename Dtype>
void ConcateSubImagesInBlobs_gpu( const Blob<Dtype>& src,
		Blob<Dtype>& dst,const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w, const int stride_h,
	    const int stride_w, const int out_img_h, const int out_img_w);
#endif

}  // namespace caffe




#endif   // CAFFE_UTIL_UTIL_IMG_H_
