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

#include "caffe/layers/fcn_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "../../3rdparty/HoG/fhog.hpp"

using namespace cv;
using namespace std;

namespace caffe {

  template <typename Dtype>
  FCNDataLayer<Dtype>::~FCNDataLayer<Dtype>() {
    this->StopInternalThread();
  }

  template <typename Dtype>
  void FCNDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                       const vector<Blob<Dtype>*>& top) {
    ImageDataParameter image_data_param = this->layer_param_.image_data_param();
    const int new_height = image_data_param.new_height();
    const int new_width = image_data_param.new_width();
    const bool is_color = image_data_param.is_color();
    string root_folder = image_data_param.root_folder();
    FCNDataParameter fcn_data_param = this->layer_param_.fcn_data_param();
    template_size = cv::Size(fcn_data_param.template_w(), fcn_data_param.template_h());
    expand_left = fcn_data_param.expand_left();
    expand_right = fcn_data_param.expand_right();
    expand_top = fcn_data_param.expand_top();
    expand_bottom = fcn_data_param.expand_bottom();
    roi_multiply = cv::Size(fcn_data_param.roi_multiply_w(), fcn_data_param.roi_multiply_h());
    scale_step = fcn_data_param.scale_step();
    scale_step_num = fcn_data_param.scale_step_num();
    gaussian_size = cv::Size(fcn_data_param.gaussian_size_w(), fcn_data_param.gaussian_size_h());
    gaussian_std_h = fcn_data_param.gaussian_std_h();
    gaussian_std_w = fcn_data_param.gaussian_std_w();
    use_hog = fcn_data_param.use_hog();
    hog_cell_size = fcn_data_param.hog_cell_size();

    CHECK((new_height == 0 && new_width == 0) ||
          (new_height > 0 && new_width > 0)) << "Current implementation requires "
          "new_height and new_width to be set at the same time.";
    // Read the file with filenames and labels
    const string& source = image_data_param.source();
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    string image_folder;
    string filename;
    double x, y, w, h;
    int imageId;

    while (!infile.eof()) {
      infile >> image_folder;
      image_folder_list.push_back(image_folder);
    }

    for (auto folder : image_folder_list) {
      ifstream image_rect_label(root_folder + folder + "\\label.txt");
      CHECK(!image_rect_label.fail()) << "label.txt do not exist!";
      while (!image_rect_label.eof()) {
        image_rect_label >> filename >> x >> y >> w >> h;
        if (x > 0 && y > 0 && w > 10 && h > 10)
          image_rect_list.push_back(make_pair(root_folder + folder + "\\" + filename, Rect2d(x, y, w, h)));
      }
    }

    if (image_data_param.shuffle()) {
      // randomly shuffle data
      LOG(INFO) << "Shuffling data";
      const unsigned int prefetch_rng_seed = caffe_rng_rand();
      prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
      ShuffleImages();
    }
    LOG(INFO) << "A total of " << image_rect_list.size() << " images.";

    image_id = 0;
    // Check if we would need to randomly skip a few data points
    if (image_data_param.rand_skip()) {
      unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points.";
      CHECK_GT(image_rect_list.size(), skip) << "Not enough points to skip";
      image_id = skip;
    }
    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img = ReadImageToCVMat(image_rect_list[image_id].first,
                                      new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << image_rect_list[image_id].first;

    target_temp = Mat::zeros(gaussian_size, CV_32FC1);
    target_temp.at<float>(Point(gaussian_size.width / 2, gaussian_size.height/2)) = 1.0f;
    cv::GaussianBlur(target_temp, target_temp, gaussian_size, gaussian_std_w, gaussian_std_h);
    target_temp /= target_temp.at<float>(Point(gaussian_size.width / 2, gaussian_size.height / 2));
    target_wheel_size = Size2d((double)template_size.width / (1.0 + expand_left + expand_right),
      (double)template_size.height / (1 + expand_top + expand_bottom));
    target_roi_size = Size(template_size.width * roi_multiply.width, template_size.height * roi_multiply.height);
    target_featuremap_size = target_roi_size;
    if (use_hog) {
      target_featuremap_size.width /= hog_cell_size;
      target_featuremap_size.height /= hog_cell_size;
    }
    target_heatmap_size = Size(target_featuremap_size.width - template_size.width + 1, target_featuremap_size.height - template_size.height + 1);

    // Use data_transformer to infer the expected blob shape from a cv_image.
    vector<int> top_shape = { 1, use_hog ? 31 : 1, target_featuremap_size.height, target_featuremap_size.width };
    this->transformed_data_.Reshape(top_shape);
    // Reshape prefetch_data and top[0] according to the batch_size.
    const int batch_size = image_data_param.batch_size();
    CHECK_GT(batch_size, 0) << "Positive batch size required";
    top_shape[0] = batch_size * scale_step_num;
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(top_shape);
    }
    top[0]->Reshape(top_shape);

    LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
    // label
    vector<int> label_shape = { batch_size * scale_step_num, 2, target_heatmap_size.height, target_heatmap_size.width };
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }

  template <typename Dtype>
  void FCNDataLayer<Dtype>::ShuffleImages() {
    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(image_rect_list.begin(), image_rect_list.end(), prefetch_rng);
  }

  // This function is called on prefetch thread
  template <typename Dtype>
  void FCNDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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
    FHoG hog;

    // Reshape according to the first image of each batch
    // on single input batches allows for inputs of varying dimension.
    cv::Mat cv_img = ReadImageToCVMat(image_rect_list[image_id].first,
      0, 0, is_color);
    CHECK(cv_img.data) << "Could not load " << image_rect_list[image_id].first;
    // Use data_transformer to infer the expected blob shape from a cv_img.
    vector<int> top_shape = { 1, use_hog ? 31 : 1, target_featuremap_size.height, target_featuremap_size.width };
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size * scale_step_num;
    batch->data_.Reshape(top_shape);
    vector<int> label_shape = { batch_size * scale_step_num, 2, target_heatmap_size.height, target_heatmap_size.width };
    batch->label_.Reshape(label_shape);

    Dtype* prefetch_data = batch->data_.mutable_cpu_data();
    Dtype* prefetch_label = batch->label_.mutable_cpu_data();

    // datum scales
    const int total_image = image_rect_list.size();
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      // get a blob
      timer.Start();
      CHECK_GT(total_image, image_id);
      cv::Mat image;
      image = ReadImageToCVMat(image_rect_list[image_id].first,
        0, 0, is_color);
      CHECK(image.data) << "Could not load " << image_rect_list[image_id].first;
      read_time += timer.MicroSeconds();
      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      Mat target_input_image;
      Mat target_map;
      {
        Rect2d wheel_rect = image_rect_list[image_id].second;
        double refined_h = wheel_rect.width / target_wheel_size.width * target_wheel_size.height;
        wheel_rect.y += (wheel_rect.height - refined_h) / 2;
        wheel_rect.height = refined_h;
        double base_scale = wheel_rect.height / target_wheel_size.height;
        int all_non_zero_count = 0;
        for (int s = -(scale_step_num - 1) / 2; s <= (scale_step_num - 1) / 2; s++) {
          double scale_factor = pow(scale_step, s);
          double scale = base_scale * scale_factor;
          Rect2d scaled_wheel_rect = wheel_rect;
          scaled_wheel_rect.x += wheel_rect.width * (1 - scale_factor) / 2;
          scaled_wheel_rect.y += wheel_rect.height * (1 - scale_factor) / 2;
          scaled_wheel_rect.width = wheel_rect.width * scale_factor;
          scaled_wheel_rect.height = wheel_rect.height * scale_factor;
          Rect2d expanded_scaled_wheel_rect = Rect2d(scaled_wheel_rect.x - scaled_wheel_rect.width * expand_left,
            scaled_wheel_rect.y - scaled_wheel_rect.height * expand_top,
            scaled_wheel_rect.width * (1 + expand_left + expand_right),
            scaled_wheel_rect.height * (1 + expand_top + expand_bottom));

          Rect integer_template_rect = Rect(floor(expanded_scaled_wheel_rect.x + 0.5), floor(expanded_scaled_wheel_rect.y + 0.5),
            floor(expanded_scaled_wheel_rect.width + 0.5), floor(expanded_scaled_wheel_rect.height + 0.5));
          Rect2d ROI = Rect2d(expanded_scaled_wheel_rect.x - (roi_multiply.width - 1) / 2 * expanded_scaled_wheel_rect.width,
            expanded_scaled_wheel_rect.y - (roi_multiply.height - 1) / 2 * expanded_scaled_wheel_rect.height,
            expanded_scaled_wheel_rect.width * roi_multiply.width,
            expanded_scaled_wheel_rect.height * roi_multiply.height);
          if (ROI.x < 0) {
            ROI.width -= ROI.x;
            ROI.x = 0;
          }
          if (ROI.y < 0) {
            ROI.height -= ROI.y;
            ROI.y = 0;
          }
          if (ROI.x + ROI.width > image.cols - 1) {
            ROI.x -= ROI.x + ROI.width - image.cols + 1;
          }
          if (ROI.y + ROI.height > image.rows - 1) {
            ROI.y -= ROI.y + ROI.height - image.rows + 1;
          }
          ROI.x = floor(ROI.x + 0.5);
          ROI.y = floor(ROI.y + 0.5);
          ROI.width = floor(ROI.width + 0.5);
          ROI.height = floor(ROI.height + 0.5);
          Rect2d wheel_wrt_roi = expanded_scaled_wheel_rect;
          wheel_wrt_roi.x -= ROI.x;
          wheel_wrt_roi.y -= ROI.y;

          resize(image(ROI), target_input_image, target_roi_size);
          double image_scale_rate = target_roi_size.width / ROI.width;
          Rect2d template_ground_truth = wheel_wrt_roi;
          template_ground_truth.x *= image_scale_rate / hog_cell_size;
          template_ground_truth.y *= image_scale_rate / hog_cell_size;
          template_ground_truth.width *= image_scale_rate / hog_cell_size;
          template_ground_truth.height *= image_scale_rate / hog_cell_size;

          target_map = Mat::zeros(target_heatmap_size, CV_32FC1);
          target_map.at<float>(Point(template_ground_truth.x, template_ground_truth.y)) = 1.0f;
          cv::GaussianBlur(target_map, target_map, gaussian_size, gaussian_std_w, gaussian_std_h);
          double ideal_max_value = target_temp.at<float>(Point(gaussian_size.width / 2, gaussian_size.height / 2 + s));
          double max_val, min_val;
          cv::minMaxLoc(target_map, &min_val, &max_val);
          target_map *= ideal_max_value / max_val;
          int offset_data = batch->data_.offset(item_id * scale_step_num + s + (scale_step_num - 1) / 2);
          this->transformed_data_.set_cpu_data(prefetch_data + offset_data);
          if (use_hog) {
            target_input_image.convertTo(target_input_image, CV_32FC1);
            vector<Mat> hog_feature = hog.extract(target_input_image, 2, hog_cell_size, 9);
            CHECK_EQ(hog_feature.size(), top_shape[1]) << "Please check hog feature channels.";
            CHECK_EQ(hog_feature[0].rows, top_shape[2])<< "Please check hog feature height.";
            CHECK_EQ(hog_feature[0].cols, top_shape[3]) << "Please check hog feature width.";
            for (int c = 0; c < hog_feature.size(); c++) {
              for (int i = 0; i < hog_feature[0].cols; i++) {
                for (int j = 0; j < hog_feature[0].rows; j++) {
                  this->transformed_data_.mutable_cpu_data()[target_map.cols * target_map.rows * c + target_map.cols *j + i]
                    = hog_feature[c].at<float>(Point(i, j));
                }
              }
            }
          }
          else {
            this->data_transformer_->Transform(target_input_image, &(this->transformed_data_));
          }

          int offset_label = batch->label_.offset(item_id * scale_step_num + s + (scale_step_num - 1) / 2);
          for (int i = 0; i < target_map.cols; i++) {
            for (int j = 0; j < target_map.rows; j++) {
              float value = target_map.at<float>(Point(i, j));
              prefetch_label[offset_label + target_map.cols *j + i] = value;
              if (value > 0.5) {
                prefetch_label[offset_label + target_map.cols * target_map.rows + target_map.cols *j + i] = Dtype(1.0);
                all_non_zero_count++;
              }
              else {
                prefetch_label[offset_label + target_map.cols * target_map.rows + target_map.cols *j + i] = Dtype(0.0);
              }
            }
          }
        }
        for (int s = -(scale_step_num - 1) / 2; s <= (scale_step_num - 1) / 2; s++) {
          int offset_label = batch->label_.offset(item_id * scale_step_num + s + (scale_step_num - 1) / 2);
          for (int i = 0; i < target_map.cols; i++) {
            for (int j = 0; j < target_map.rows; j++) {
              float value = prefetch_label[offset_label + target_map.cols *j + i];
              if (prefetch_label[offset_label + target_map.cols * target_map.rows + target_map.cols *j + i] < Dtype(0.01)) {
                prefetch_label[offset_label + target_map.cols * target_map.rows + target_map.cols *j + i] = (Dtype)all_non_zero_count / (Dtype)(target_map.cols * target_map.rows * scale_step_num - all_non_zero_count);
              }
            }
          }
        }
      }
      trans_time += timer.MicroSeconds();

      // go to the next iter
      image_id++;
      if (image_id >= total_image) {
        // We have reached the end. Restart from the first.
        LOG(INFO) << "Restarting data prefetching from start.";
        image_id = 0;
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

  INSTANTIATE_CLASS(FCNDataLayer);
  REGISTER_LAYER_CLASS(FCNData);

}  // namespace caffe
#endif  // USE_OPENCV
