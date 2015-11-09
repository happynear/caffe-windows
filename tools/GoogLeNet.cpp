// GoogLeNet.cpp : 定义控制台应用程序的入口点。
//

#include "caffe/caffe.hpp"
#include "opencv2/opencv.hpp"
#include "gflags\gflags.h"
#include "direct.h"

using namespace caffe;
using namespace cv;
using namespace std;

DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_string(imagefile, "",
	"The image file path.");
DEFINE_string(labelfile, "",
	"The label file path.");

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
	
	Datum datum;
	LOG(INFO)<<"reading model from "<<FLAGS_model;
 	Net<float> caffe_test_net(FLAGS_model,TEST);
	LOG(INFO)<<"reading weights from "<<FLAGS_weights;
 	caffe_test_net.CopyTrainedLayersFrom(FLAGS_weights);
	LOG(INFO)<<"reading label info from"<<FLAGS_labelfile;
	NetParameter net_param;
	// For intermediate results, we will also dump the gradient values.
	caffe_test_net.ToProto(&net_param, false);
	char iter_str_buffer[256];
	sprintf_s(iter_str_buffer, 256, "thinned_net.caffemodel");
	LOG(INFO) << "Snapshotting to " << iter_str_buffer;
	WriteProtoToBinaryFile(net_param, iter_str_buffer);
	ifstream labels_file(FLAGS_labelfile,ios_base::in);
	vector<string> labels;
	char *buf = new char[256];
	while(!labels_file.eof())
	{
		labels_file.getline(buf,256);
		labels.push_back(string(buf));
	}

	//cv::Mat image = cv::imread(FLAGS_imagefile, 1);
	int W=256,H=256;
	//resize(image,image,cv::Size(W,H));
	////image = image.t();
	//cv::Mat imgFloat;
	//image.convertTo(imgFloat,CV_32F);//,0.00390625
	ReadImageToDatum(FLAGS_imagefile,1,W,H,1,&datum);
	vector<Datum> datum_vector;
	datum_vector.push_back(datum);
	MemoryDataLayer<float>* data_layer_ptr = (MemoryDataLayer<float>*)&(*caffe_test_net.layers()[0]);
	data_layer_ptr->AddDatumVector(datum_vector);
	const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled();
	for(int i=0;i<5;i++)
	{
		LOG(ERROR)<<"top"<<i<<":"<< labels[result[0]->cpu_data()[i]]<<" "<<result[0]->cpu_data()[i+5];
	}
	//test_accuracy+=result[0]->cpu_data()[0];

	return 0;
}

