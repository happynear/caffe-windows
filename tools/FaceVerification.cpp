// GoogLeNet.cpp : 定义控制台应用程序的入口点。
//

#include "caffe/caffe.hpp"
#include "opencv2/opencv.hpp"
#include "gflags\gflags.h"
#include "direct.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

using namespace caffe;
using namespace cv;
using namespace std;
using namespace dlib;

DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");

void initGlog()
{
	FLAGS_log_dir=".\\log\\";
	_mkdir(FLAGS_log_dir.c_str());
	std::string LOG_INFO_FILE;
	std::string LOG_WARNING_FILE;
	std::string LOG_ERROR_FILE;
	std::string LOG_FATAL_FILE;
	std::string now_time=boost::posix_time::to_iso_extended_string(boost::posix_time::second_clock::local_time());
	now_time[13]='-';
	now_time[16]='-';
	LOG_INFO_FILE = FLAGS_log_dir + "INFO" + now_time + ".txt";
	google::SetLogDestination(google::GLOG_INFO,LOG_INFO_FILE.c_str());
	LOG_WARNING_FILE = FLAGS_log_dir + "WARNING" + now_time + ".txt";
	google::SetLogDestination(google::GLOG_WARNING,LOG_WARNING_FILE.c_str());
	LOG_ERROR_FILE = FLAGS_log_dir + "ERROR" + now_time + ".txt";
	google::SetLogDestination(google::GLOG_ERROR,LOG_ERROR_FILE.c_str());
	LOG_FATAL_FILE = FLAGS_log_dir + "FATAL" + now_time + ".txt";
	google::SetLogDestination(google::GLOG_FATAL,LOG_FATAL_FILE.c_str());
}

Mat FaceAlignment(string facefile, frontal_face_detector& detector,shape_predictor& sp )
{
	cv::Mat temp = cv::imread(facefile);
	if (!temp.data) return Mat::zeros(100,100,CV_32F);
	cv::Mat temp_gray = temp.clone();
	cv::cvtColor(temp_gray,temp_gray,CV_BGR2GRAY);
	dlib::cv_image<bgr_pixel> cv_img(temp);
			
	array2d<rgb_pixel> img;
	assign_image(img,cv_img);
	std::vector<dlib::rectangle> dets = detector(img);
	if(dets.size()>1)
	{
		int maxArea = 0,p = 0;
		for (int i = 0;i<dets.size();i++)
		{
			double vote = dets[i].width()*dets[i].height();
			vote -= abs(temp_gray.cols/2 - (dets[i].left() + dets[i].right())/2) * abs(temp_gray.rows/2 - (dets[i].top() + dets[i].bottom())/2);
			if(vote>maxArea)
			{
				maxArea = dets[i].width()*dets[i].height();
				p = i;
			}
		}
		if(p!=0)
		{
			cout<<"choose "<<p<"-th face\n";
			dets.erase(dets.begin(),dets.begin()+p-1);
		}
	}
	std::vector<full_object_detection> shapes;
	if(dets.size()>0)
	{
        full_object_detection shape = sp(img, dets[0]);
        shapes.push_back(shape);
    }
	else return Mat::zeros(100,100,CV_32F);
	if(shapes.size()>0)
	{
		dlib::array<array2d<rgb_pixel> > face_chips;
		extract_image_chips(img, get_face_chip_details(shapes,100UL,0.5785), face_chips);
			
		array2d<unsigned char> gray_face;
		assign_image(gray_face,face_chips[0]);
	
		Mat face = toMat (gray_face);
		return face.clone();
		//win_faces.set_image(tile_images(face_chips));
	}
	else return Mat::zeros(100,100,CV_32F);
}

int main(int argc, char* argv[])
{
	FLAGS_alsologtostderr = 1;
	caffe::GlobalInit(&argc, &argv);
	initGlog();
	 // Set device id and mode
	if (FLAGS_gpu >= 0) {
		LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
		Caffe::SetDevice(FLAGS_gpu);
		Caffe::set_mode(Caffe::GPU);
	} else {
		LOG(INFO) << "Use CPU.";
		Caffe::set_mode(Caffe::CPU);
	}
	
	LOG(INFO)<<"reading model from "<<FLAGS_model;
 	Net<float> caffe_test_net(FLAGS_model,TEST);
	LOG(INFO)<<"reading weights from "<<FLAGS_weights;
 	caffe_test_net.CopyTrainedLayersFrom(FLAGS_weights);
	
	frontal_face_detector detector = get_frontal_face_detector();
    // And we also need a shape_predictor.  This is the tool that will predict face
    // landmark positions given an image and face bounding box.  Here we are just
    // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
    // as a command line argument.
	char landmark_dat[256]="shape_predictor_68_face_landmarks.dat";
    shape_predictor sp;
    deserialize(landmark_dat) >> sp;

	//cv::Mat image = cv::imread(FLAGS_imagefile, 1);
	int W=100,H=100;
	//resize(image,image,cv::Size(W,H));
	////image = image.t();
	//cv::Mat imgFloat;
	//image.convertTo(imgFloat,CV_32F);//,0.00390625
	Mat face1,face2;
	face1 = FaceAlignment(argv[1],detector,sp);
	face2 = FaceAlignment(argv[2],detector,sp);
	imshow("face1",face1);
	imshow("face2",face2);
	waitKey();
	std::vector<Mat> datum_vector;
	datum_vector.push_back(face1);
	datum_vector.push_back(face2);
	std::vector<int> labels;
	labels.push_back(1);
	labels.push_back(2);
	MemoryDataLayer<float>* data_layer_ptr = (MemoryDataLayer<float>*)&(*caffe_test_net.layers()[0]);
	data_layer_ptr->AddMatVector(datum_vector,labels);
	const std::vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled();
	cout<<"相似度:"<<1 - result[0]->cpu_data()[0]<<endl;
	
	//test_accuracy+=result[0]->cpu_data()[0];

	return 0;
}

