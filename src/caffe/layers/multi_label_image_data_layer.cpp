#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <sstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <random>

#include "caffe/data_layers.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
MultiLabelImageDataLayer<Dtype>::~MultiLabelImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

typedef std::mt19937 RANDOM_ENGINE;

template <typename Dtype>
void extract_face(cv::Mat& input_image, shared_ptr<vector<Dtype> > points, int point_count,
                  int new_width, int new_height,
                  float max_shear_ratio = 0, float max_aspect_ratio=0, float max_rotate_angle = 0,
                  float min_random_scale = 1, float max_random_scale = 1) {
  LOG(INFO) << max_shear_ratio << " " << max_aspect_ratio << " " << max_rotate_angle << " " << min_random_scale << " " << max_random_scale;
  cv::Point2d face_center;
  face_center.x = (*points)[0] + (*points)[2] - (*points)[4];
  face_center.y = (((*points)[1] + (*points)[3]) / 2 + (*points)[5]) / 2;
  double eye_scale = std::max((*points)[4] - (*points)[0], (*points)[2] - (*points)[4]) * 2 * 1.5;

  RANDOM_ENGINE prnd(time(NULL));
  std::uniform_real_distribution<float> rand_uniform(0, 1);
  // shear
  float s = rand_uniform(prnd) * max_shear_ratio * 2 - max_shear_ratio;
  // rotate
  int angle = std::uniform_int_distribution<int>(
    -max_rotate_angle, max_rotate_angle)(prnd);
  float a = cos(angle / 180.0 * CV_PI);
  float b = sin(angle / 180.0 * CV_PI);
  // scale
  float scale = rand_uniform(prnd) *
    (max_random_scale - min_random_scale) + min_random_scale;
  scale = scale * new_height / (eye_scale * 2);
  // aspect ratio
  float ratio = rand_uniform(prnd) *
    max_aspect_ratio * 2 - max_aspect_ratio + 1;
  float hs = 2 * scale / (1 + ratio);
  float ws = ratio * hs;
  int flip = std::uniform_int_distribution<int>(0, 1)(prnd)* 2 - 1;
  hs *= flip;

  cv::Mat M(2, 3, CV_32F);
  M.at<float>(0, 0) = hs * a - s * b * ws;
  M.at<float>(1, 0) = -b * ws;
  M.at<float>(0, 1) = hs * b + s * a * ws;
  M.at<float>(1, 1) = a * ws;
  M.at<float>(0, 2) = new_width / 2 -M.at<float>(0, 0) * face_center.x - M.at<float>(0, 1) * face_center.y;
  M.at<float>(1, 2) = new_height / 2 -M.at<float>(1, 0) * face_center.x - M.at<float>(1, 1) * face_center.y;
  LOG(INFO) << M.at<float>(0, 0) << " " << M.at<float>(1, 0) << " " << M.at<float>(0, 1) << " " << M.at<float>(1, 1) << " " << new_width << " " << new_height << " " << flip;
  cv::Mat temp_;
  cv::warpAffine(input_image, temp_, M, cv::Size(new_width, new_height),
                 cv::INTER_LINEAR,
                 cv::BORDER_TRANSPARENT,
                 cv::Scalar(123.68, 116.779, 103.939));
  input_image = temp_.clone();
  for (int j = 0; j < point_count; j++) {
    Dtype x = M.at<float>(0, 0)*(*points)[j * 2] + M.at<float>(0, 1) * (*points)[j * 2 + 1] + M.at<float>(0, 2);
    Dtype y = M.at<float>(1, 0)*(*points)[j * 2] + M.at<float>(1, 1) * (*points)[j * 2 + 1] + M.at<float>(1, 2);
    (*points)[j * 2] = x;
    (*points)[j * 2 + 1] = y;
  }
  if (flip == -1) {
    std::swap((*points)[0], (*points)[2]);
    std::swap((*points)[1], (*points)[3]);
    std::swap((*points)[6], (*points)[8]);
    std::swap((*points)[7], (*points)[9]);
  }
}

template <typename Dtype>
void MultiLabelImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = image_data_param.source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  char this_line[1024];
  label_count = 0;
  
  while (!infile.eof()) {
    infile.getline(this_line, 1024);
    std::stringstream stream;
    stream << this_line;
    stream >> filename;
    Dtype label;
    shared_ptr<vector<Dtype> > labels_ptr(new vector<Dtype>);
    while (!stream.eof()) {
      stream >> label;
      labels_ptr->push_back(label);
    }
    if (label_count == 0) {
      label_count = labels_ptr->size();
      LOG(INFO) << "num of classifiers: " << label_count;
    }
    else {
      CHECK_EQ(label_count, labels_ptr->size()) << "label count do not match for file:" << filename;
    }
    lines_.push_back(std::make_pair(filename, labels_ptr));
  }

  if (image_data_param.shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (image_data_param.rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img;
  if (image_data_param.face_transform()) {
    cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                              0, 0, is_color);
    extract_face(cv_img, lines_[lines_id_].second, image_data_param.face_point_num(),
                 image_data_param.new_width(), image_data_param.new_height(),
                 image_data_param.max_shear_ratio(), image_data_param.max_aspect_ratio(), image_data_param.max_rotate_angle(),
                 image_data_param.min_random_scale(), image_data_param.max_random_scale());
  }
  else {
    cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                              new_height, new_width, is_color);
  }
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = image_data_param.batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape = { batch_size, label_count };
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void MultiLabelImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void MultiLabelImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img;
  if (image_data_param.face_transform()) {
    cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                              0, 0, is_color);
    extract_face(cv_img, lines_[lines_id_].second, image_data_param.face_point_num(),
                 image_data_param.new_width(), image_data_param.new_height(),
                 image_data_param.max_shear_ratio(), image_data_param.max_aspect_ratio(), image_data_param.max_rotate_angle(),
                 image_data_param.min_random_scale(), image_data_param.max_random_scale());
  }
  else {
    cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                              new_height, new_width, is_color);
  }
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img;
    if (image_data_param.face_transform()) {
      cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                0, 0, is_color);
    }
    else {
      cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                        new_height, new_width, is_color);
    }
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    if (image_data_param.face_transform()) {
      extract_face(cv_img, lines_[lines_id_].second, image_data_param.face_point_num(),
                   image_data_param.new_width(), image_data_param.new_height(),
                   image_data_param.max_shear_ratio(), image_data_param.max_aspect_ratio(), image_data_param.max_rotate_angle(),
                   image_data_param.min_random_scale(), image_data_param.max_random_scale());
    }
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    for (int label_id = 0; label_id < label_count; ++label_id)
    {
      prefetch_label[item_id * label_count + label_id] = (*lines_[lines_id_].second)[label_id];
    }
    
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MultiLabelImageDataLayer);
REGISTER_LAYER_CLASS(MultiLabelImageData);

}  // namespace caffe
#endif  // USE_OPENCV
