// This script converts the MNIST dataset to a lmdb (default) or
// leveldb (--backend=leveldb) format used by caffe to load data.
// Usage:
//    convert_mnist_data [FLAGS] input_image_file input_label_file
//                        output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#if defined(USE_LEVELDB) && defined(USE_LMDB)
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#endif

#include <stdint.h>
#include <sys/stat.h>
#include <direct.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/proto/caffe.pb.h"
//#include <../../buildVS2013/convert_mnist_data/strCoding.h>


#if defined(USE_LEVELDB) && defined(USE_LMDB)

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

DEFINE_string(backend, "lmdb", "The backend for storing the result");

uint32_t swap_endian(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}



void convert_dataset(const char* image_filename, const char* label_filename, const char* limit,
	const char* db_path, const string& db_backend) {
	// Open files

	bool usesummary = atoi(limit) > 1;// strcmp(label_filename, "summary") == 0;
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file;
	int limit_num = atoi(limit);
	int start_num = atoi(label_filename);
	if (!usesummary){
		std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
		CHECK(label_file) << "Unable to open file " << label_filename;
	}


	CHECK(image_file) << "Unable to open file " << image_filename;
	
	// Read the magic and the meta data
	uint32_t magic;
	uint64_t num_items;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;
	uint32_t rows1;
	uint32_t cols1;

	if (!usesummary)
	{
		image_file.read(reinterpret_cast<char*>(&magic), 4);
		magic = swap_endian(magic);
		CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
		label_file.read(reinterpret_cast<char*>(&magic), 4);
		magic = swap_endian(magic);
		CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
		image_file.read(reinterpret_cast<char*>(&num_items), 4);
		num_items = swap_endian(num_items);
		label_file.read(reinterpret_cast<char*>(&num_labels), 4);
		num_labels = swap_endian(num_labels);
		CHECK_EQ(num_items, num_labels);
		image_file.read(reinterpret_cast<char*>(&rows), 4);
		rows = swap_endian(rows);
		image_file.read(reinterpret_cast<char*>(&cols), 4);
		cols = swap_endian(cols);
	}
	else
	{
		//start_num = atoi(label_filename);
		num_items = limit_num;
		atoi(label_filename);
		rows = 22;
		cols = 44;
		//高字节从B0-F7，低字节从A1-FE
		//176-247 ，161 - 254
		rows1 = 72;
		cols1 = 94;
	}

//title:


	// lmdb
	MDB_env *mdb_env;
	MDB_dbi mdb_dbi;
	MDB_val mdb_key, mdb_data;
	MDB_txn *mdb_txn;
	// leveldb
	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 268435456;
	leveldb::WriteBatch* batch = NULL;

	// Open db
	if (db_backend == "leveldb") {  // leveldb
		LOG(INFO) << "Opening leveldb " << db_path;
		leveldb::Status status = leveldb::DB::Open(
			options, db_path, &db);
		CHECK(status.ok()) << "Failed to open leveldb " << db_path
			<< ". Is it already existing?";
		batch = new leveldb::WriteBatch();
	}
	else if (db_backend == "lmdb") {  // lmdb
		LOG(INFO) << "Opening lmdb " << db_path;
		CHECK_EQ(_mkdir(db_path), 0)
			<< "mkdir " << db_path << "failed";
		CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
		CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
			<< "mdb_env_set_mapsize failed";
		CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
			<< "mdb_env_open failed";
		CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
			<< "mdb_txn_begin failed";
		CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
			<< "mdb_open failed. Does the lmdb already exist? ";
	}
	else {
		LOG(FATAL) << "Unknown db backend " << db_backend;
	}

	// Storing to db
	char label;
	unsigned char* tpixels = new unsigned char[rows * cols];
	unsigned char* pixels = new unsigned char[rows1 * cols1];
	int count = 0;
	const int kMaxKeyLength = 10;
	char key_cstr[kMaxKeyLength];
	string value;
	int mpos=0;
	int mpos2=0;
	int label_i = 0;
	int spos = 0;
	int spos1 = 0;
	int empty_line = 0;
	//string keyword = "大家好,欢迎你";
	//strCoding cfm;
	string Temps = "";
	string Output = "";

	Datum datum;
	datum.set_channels(1);
	datum.set_height(rows1);
	datum.set_width(cols1);
	LOG(INFO) << "A total of " << num_items << " items.";
	LOG(INFO) << "Rows: " << rows1 << " Cols: " << cols1;
	string sline;
	//char sentenc;
	for (int item_id = 0; item_id < start_num + num_items; ++item_id) {
		
		if (usesummary){
			sline.clear();
			getline(image_file, sline, '\n');
			if (item_id < start_num)
				continue;
			if (sline.length() < 2)
				continue;
			//3:1.92:
			mpos = sline.find_first_of(':');
			mpos2 = mpos;
			if (mpos > 2 || mpos <= 0)
				continue;

			if (strcmp(sline.substr(0, 5).c_str(), "title") != 0){
				try{
					label_i = stoi(sline.substr(0, mpos));
				}
				catch (int i){
					continue;
				}
				if (sline.find_first_of(':', mpos + 1) < 22); {
					mpos2 = sline.find_first_of(':', mpos + 1);
				}
			}
			else
				continue;


			 if (sline.substr(mpos2 + 1).length() < 17)
			{
				if (num_items == 99)//99999 just for testing
					LOG(INFO) << " item_id:" << item_id << " label:" << label_i << " str:" << sline.substr(mpos2 + 1) << " mpos2:" << mpos2;
				continue;
			}
			 mpos = sline.substr(mpos2 + 1).length();
			 if (mpos > rows*cols - 1)
				 mpos = rows*cols - 1;


			//cfm.UTF_8ToGB2312(Temps, (char *)sline.substr(mpos2 + 1).data(), strlen(sline.substr(mpos2 + 1).data()));
			 Temps.clear();
			 Temps = sline.substr(mpos2 + 1).data();
			if (num_items == 999)//99999 just for testing
			{
				LOG(INFO) << " sline :" << sline.substr(mpos2 + 1).data();
				//LOG(INFO) << " GB2312 Temps:" << Temps.data();
			}

			memset(tpixels, 0, rows*cols);
			memset(pixels, 0, rows1*cols1);

			memcpy(tpixels, sline.substr(mpos2 + 1).c_str(), mpos);
			spos1 = 0;
			for (spos = 0; spos < rows*cols; spos++){
				/*
				if (spos1 > 0){
					if (spos1 % 2 == 1 && tpixels[spos] >= 128)
					{
						if ((unsigned char)(tpixels[(spos - 1)]) < 128)
						{
							if (num_items == 99)//99999 just for testing
							{
								//LOG(INFO) << "tpixels[spos-1]:" << tpixels[spos - 1] << " tpixels[spos ]:" << tpixels[spos];
							}
							//pixels[spos1] = tpixels[spos - 1]; 
							spos1++;
						}
					}
				}
				
			if (tpixels[spos] & 0x80==0){
			spos++;
			if (num_items == 99)//99999 just for testing
			{
			LOG(INFO) << "tpixels[spos-1]:" << tpixels[spos - 1] << " tpixels[spos ]:" << tpixels[spos];
			}
			}

			if (spos1!=0 && spos1%cols == 0){
			empty_line++;
			spos1 += cols;
			}
			*/
				//高字节从B0-F7，低字节从A1-FE
				//176-247 ，161 - 254  // 72 x 94
				if (tpixels[spos] >= 176)
					if (tpixels[spos + 1] >= 161){
						int whigh = tpixels[spos] - 176;
						int wlow = tpixels[spos + 1] - 161;
						if (whigh >= 0 && whigh <= rows1 && wlow >= 0 && wlow <= cols1){
							if (pixels[whigh*cols1 + wlow] < 255){
								pixels[whigh*cols1 + wlow] += 1;
								spos++;
							}
						}
					}

				//pixels[spos1] = tpixels[spos];
				//mpos2 = tpixels[spos];
				spos1++;
			}

			if (num_items == 999)//99999 just for testing
			{
				memset(tpixels, 0, rows*cols);
				spos1 = 0;
				for (spos = 0; spos < rows1*cols1; spos++){
					if (pixels[spos] > 0){
						
						tpixels[spos1++] = (uint8_t)(spos / cols1 + 176);
						tpixels[spos1++] = (uint8_t)((uint8_t)(spos % cols1) + 161);
						
					}
				}
				LOG(INFO) << "spos1:" << spos1 << " tpixels:" << tpixels;
				//LOG(INFO) << " tpixels:" << (char *)(tpixels);
				//LOG(INFO) << " pixels:" << pixels;
			}

			label =(char) label_i ;

		}
		else{
			image_file.read((char *)pixels, rows * cols);
			label_file.read(&label, 1);
		}
		datum.set_data(pixels, rows1*cols1);
		datum.set_label(label);
		sprintf_s(key_cstr, kMaxKeyLength, "%09d", item_id);
		datum.SerializeToString(&value);
		string keystr(key_cstr);

		// Put in db
		if (db_backend == "leveldb") {  // leveldb
			batch->Put(keystr, value);
		}
		else if (db_backend == "lmdb") {  // lmdb
			mdb_data.mv_size = value.size();
			mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
			mdb_key.mv_size = keystr.size();
			mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
			CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
				<< "mdb_put failed";
		}
		else {
			LOG(FATAL) << "Unknown db backend " << db_backend;
		}

		if (++count % 1000 == 0) {
			// Commit txn
			if (db_backend == "leveldb") {  // leveldb
				db->Write(leveldb::WriteOptions(), batch);
				delete batch;
				batch = new leveldb::WriteBatch();
			}
			else if (db_backend == "lmdb") {  // lmdb
				CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
					<< "mdb_txn_commit failed";
				CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
					<< "mdb_txn_begin failed";
			}
			else {
				LOG(FATAL) << "Unknown db backend " << db_backend;
			}
		}
	}
	// write the last batch
	if (count % 1000 != 0) {
		if (db_backend == "leveldb") {  // leveldb
			db->Write(leveldb::WriteOptions(), batch);
			delete batch;
			delete db;
		}
		else if (db_backend == "lmdb") {  // lmdb
			CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
			mdb_close(mdb_env, mdb_dbi);
			mdb_env_close(mdb_env);
		}
		else {
			LOG(FATAL) << "Unknown db backend " << db_backend;
		}
		LOG(ERROR) << "Processed " << count << " files.";
	}
	delete[] pixels;
}

int main(int argc, char** argv) {
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("This script converts the MNIST dataset to\n"
		"the lmdb/leveldb format used by Caffe to load data.\n"
		"Usage:\n"
		"    convert_mnist_data [FLAGS] input_image_file input_label_file "
		"output_db_file\n"
		"The MNIST dataset could be downloaded at\n"
		"    http://yann.lecun.com/exdb/mnist/\n"
		"You should gunzip them after downloading,"
		"or directly use data/mnist/get_mnist.sh\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	const string& db_backend = FLAGS_backend;
	/*
	argc = 4;
	argv[1] = "F:\\website\\summary_out2";
	argv[2] = "99";
	argv[3] = "F:\\caffe-windows\\examples\\mnist\\mnist-test-leveldb";
	*/

	if (argc != 5) {
		gflags::ShowUsageWithFlagsRestrict(argv[0],
			"examples/mnist/convert_mnist_data");
	}
	else {
		google::InitGoogleLogging(argv[0]);
		convert_dataset(argv[1], argv[2], argv[3], argv[4], db_backend);
	}
	return 0;
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires LevelDB and LMDB; " <<
		"compile with USE_LEVELDB and USE_LMDB.";
}
#endif  // USE_LEVELDB and USE_LMDB
