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

#include "caffe/layers/multi_label_image_data_layer.hpp"
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
  void extract_face(cv::Mat& input_image, Dtype* points, int point_count,
                    int new_width, int new_height, int max_random_shift = 0,
                    float max_shear_ratio = 0, float max_aspect_ratio = 0, float max_rotate_angle = 0,
                    float min_random_scale = 1, float max_random_scale = 1,
                    bool face_mirror = false) {
    cv::Point2d face_center;
    face_center.x = (points[0] + points[2]) / 2;
    face_center.y = (points[1] + points[3]) / 2;
    cv::Point2d mouth_center;
    mouth_center.x = (points[6] + points[8]) / 2;
    mouth_center.y = (points[7] + points[9]) / 2;
    double face_scale = 2 * sqrt((face_center.x - mouth_center.x) * (face_center.x - mouth_center.x)
                                 + (face_center.y - mouth_center.y) * (face_center.y - mouth_center.y));
    RANDOM_ENGINE prnd(time(NULL));
    face_center.x += std::uniform_int_distribution<int>(-max_random_shift, max_random_shift)(prnd);
    face_center.y += std::uniform_int_distribution<int>(-max_random_shift, max_random_shift)(prnd);
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
    scale = scale * new_height / (face_scale * 2);
    // aspect ratio
    float ratio = rand_uniform(prnd) *
      max_aspect_ratio * 2 - max_aspect_ratio + 1;
    float hs = 2 * scale / (1 + ratio);
    float ws = ratio * hs;
    int flip = 1;
    if (face_mirror) {
      flip = std::uniform_int_distribution<int>(0, 1)(prnd)* 2 - 1;
    }
    hs *= flip;

    cv::Mat M(2, 3, CV_32F);
    M.at<float>(0, 0) = hs * a - s * b * ws;
    M.at<float>(1, 0) = -b * ws;
    M.at<float>(0, 1) = hs * b + s * a * ws;
    M.at<float>(1, 1) = a * ws;
    M.at<float>(0, 2) = new_width / 2 - M.at<float>(0, 0) * face_center.x - M.at<float>(0, 1) * face_center.y;
    M.at<float>(1, 2) = new_height / 2 - M.at<float>(1, 0) * face_center.x - M.at<float>(1, 1) * face_center.y;
    //LOG(INFO) << M.at<float>(0, 0) << " " << M.at<float>(1, 0) << " " << M.at<float>(0, 1) << " " << M.at<float>(1, 1) << " " << new_width << " " << new_height << " " << flip;
    cv::Mat temp_;
    cv::warpAffine(input_image, temp_, M, cv::Size(new_width, new_height),
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(128));
    for (int j = 0; j < point_count; j++) {
      Dtype x = M.at<float>(0, 0)*points[j * 2] + M.at<float>(0, 1) * points[j * 2 + 1] + M.at<float>(0, 2);
      Dtype y = M.at<float>(1, 0)*points[j * 2] + M.at<float>(1, 1) * points[j * 2 + 1] + M.at<float>(1, 2);
      points[j * 2] = x - new_width / 2;
      points[j * 2 + 1] = y - new_height / 2;
    }
    if (flip == -1) {
      std::swap(points[0], points[2]);
      std::swap(points[1], points[3]);
      std::swap(points[6], points[8]);
      std::swap(points[7], points[9]);
    }

    //temp_ = temp_(cv::Rect(crop_x, crop_y, new_width, new_height));
    input_image = temp_.clone();
  }

  template <typename Dtype>
  void MultiLabelImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                       const vector<Blob<Dtype>*>& top) {
    ImageDataParameter image_data_param = this->layer_param_.image_data_param();
    const int new_height = image_data_param.new_height();
    const int new_width = image_data_param.new_width();
    const bool is_color = image_data_param.is_color();
    string root_folder = image_data_param.root_folder();
    balance_ = image_data_param.balance_class();
    balance_by_ = image_data_param.balance_by();
    label_cut_start_ = image_data_param.label_cut_start();
    label_cut_end_ = image_data_param.label_cut_start();

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
    string last_filename = "";
    int max_label = 0;

    while (!infile.eof()) {
      infile.getline(this_line, 1024);
      std::stringstream stream;
      stream << this_line;
      stream >> filename;
      if (filename.length() < 3) break;
      if (filename == last_filename) break;
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
      last_filename = filename;
      if ((*labels_ptr)[balance_by_] > max_label) max_label = (*labels_ptr)[balance_by_];
    }

    if (balance_) {
      num_samples_ = vector<int>(max_label + 1);
      filename_by_class_ = vector<vector<std::pair<std::string, shared_ptr<vector<Dtype> > > > >(max_label + 1);
      for (auto& l : lines_) {
        num_samples_[(*l.second)[balance_by_]]++;
        filename_by_class_[(*l.second)[balance_by_]].push_back(l);
      }
      class_id_ = 0;
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
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                      new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    // Use data_transformer to infer the expected blob shape from a cv_image.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
    // Reshape prefetch_data and top[0] according to the batch_size.
    const int batch_size = image_data_param.batch_size();
    CHECK_GT(batch_size, 0) << "Positive batch size required";
    top_shape[0] = batch_size;
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->data_.Reshape(top_shape);
    }
    top[0]->Reshape(top_shape);

    LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
    // label
    vector<int> label_shape = { batch_size - label_cut_start_ - label_cut_end_, label_count };
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
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
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                      new_height, new_width, is_color);
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
      bool valid_sample = false;
      while (!valid_sample) {
        cv::Mat cv_img;
        if (image_data_param.face_transform()) {
          cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    0, 0, is_color);
        }
        else {
          cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
        }
        if (!cv_img.data) {
          LOG(INFO) << "Could not load " << lines_[lines_id_].first;
          valid_sample = false;
        }
        else{
          read_time += timer.MicroSeconds();
          timer.Start();
          if (item_id >= label_cut_start_ && item_id < batch_size - label_cut_end_) {
            for (int label_id = 0; label_id < label_count; ++label_id) {
              prefetch_label[(item_id - label_cut_start_) * label_count + label_id] = (*lines_[lines_id_].second)[label_id];
            }
          }
          // Apply transformations (mirror, crop...) to the image
          if (image_data_param.face_transform()) {
            extract_face(cv_img, &prefetch_label[item_id * label_count], image_data_param.face_point_num(),
                         image_data_param.new_width(), image_data_param.new_height(), image_data_param.max_random_shift(),
                         image_data_param.max_shear_ratio(), image_data_param.max_aspect_ratio(), image_data_param.max_rotate_angle(),
                         image_data_param.min_random_scale(), image_data_param.max_random_scale(),
                         image_data_param.face_mirror());
          }
          int offset = batch->data_.offset(item_id);
          this->transformed_data_.set_cpu_data(prefetch_data + offset);
          this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
          trans_time += timer.MicroSeconds();

          valid_sample = true;
          if (image_data_param.face_transform()) {
            for (int point_id = 0; point_id < image_data_param.face_point_num(); ++point_id) {
              if (prefetch_label[item_id * label_count + point_id * 2] < -0.45 * new_width || prefetch_label[item_id * label_count + point_id * 2] > 0.45 * new_width
                  || prefetch_label[item_id * label_count + point_id * 2 + 1] < -0.45 * new_height || prefetch_label[item_id * label_count + point_id * 2 + 1] > 0.45 * new_height) {
                valid_sample = false;
              }
            }
            if (!valid_sample) {
              LOG(INFO) << "skip " << lines_[lines_id_].first;
            }
          }
        }
        
        // go to the next iter
        lines_id_++;
        if (lines_id_ >= lines_size) {
          // We have reached the end. Restart from the first.
          LOG(INFO) << "Restarting data prefetching from start.";
          lines_id_ = 0;
          if (this->layer_param_.image_data_param().shuffle()) {
            ShuffleImages();
          }
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
