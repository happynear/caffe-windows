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
#include "stdlib.h"
#include "direct.h"
#include "string.h"
#include "string"
#include "io.h"
#include "stdio.h" 
#include <vector>
#include "iostream"
using namespace std;

class CBrowseDir
{
protected:
	//存放初始目录的绝对路径，以'\'结尾
	char m_szInitDir[_MAX_PATH];

public:
	//缺省构造器
	CBrowseDir();

	//设置初始目录为dir，如果返回false，表示目录不可用
	bool SetInitDir(const char *dir);

	//开始遍历初始目录及其子目录下由filespec指定类型的文件
	//filespec可以使用通配符 * ?，不能包含路径。
	//如果返回false，表示遍历过程被用户中止
	bool BeginBrowse(const char *filespec);
	vector<string> BeginBrowseFilenames(const char *filespec);

protected:
	//遍历目录dir下由filespec指定的文件
	//对于子目录,采用迭代的方法
	//如果返回false,表示中止遍历文件
	bool BrowseDir(const char *dir, const char *filespec);
	vector<string> GetDirFilenames(const char *dir, const char *filespec);
	//函数BrowseDir每找到一个文件,就调用ProcessFile
	//并把文件名作为参数传递过去
	//如果返回false,表示中止遍历文件
	//用户可以覆写该函数,加入自己的处理代码
	virtual bool ProcessFile(const char *filename);

	//函数BrowseDir每进入一个目录,就调用ProcessDir
	//并把正在处理的目录名及上一级目录名作为参数传递过去
	//如果正在处理的是初始目录,则parentdir=NULL
	//用户可以覆写该函数,加入自己的处理代码
	//比如用户可以在这里统计子目录的个数
	virtual void ProcessDir(const char *currentdir, const char *parentdir);
};

CBrowseDir::CBrowseDir()
{
	//用当前目录初始化m_szInitDir
	getcwd(m_szInitDir, _MAX_PATH);

	//如果目录的最后一个字母不是'\',则在最后加上一个'\'
	int len = strlen(m_szInitDir);
	if (m_szInitDir[len - 1] != '\\')
		strcat(m_szInitDir, "\\");
}

bool CBrowseDir::SetInitDir(const char *dir)
{
	//先把dir转换为绝对路径
	if (_fullpath(m_szInitDir, dir, _MAX_PATH) == NULL)
		return false;

	//判断目录是否存在
	if (_chdir(m_szInitDir) != 0)
		return false;

	//如果目录的最后一个字母不是'\',则在最后加上一个'\'
	int len = strlen(m_szInitDir);
	if (m_szInitDir[len - 1] != '\\')
		strcat(m_szInitDir, "\\");

	return true;
}

vector<string> CBrowseDir::BeginBrowseFilenames(const char *filespec)
{
	ProcessDir(m_szInitDir, NULL);
	return GetDirFilenames(m_szInitDir, filespec);
}

bool CBrowseDir::BeginBrowse(const char *filespec)
{
	ProcessDir(m_szInitDir, NULL);
	return BrowseDir(m_szInitDir, filespec);
}

bool CBrowseDir::BrowseDir(const char *dir, const char *filespec)
{
	_chdir(dir);

	//首先查找dir中符合要求的文件
	long hFile;
	_finddata_t fileinfo;
	if ((hFile = _findfirst(filespec, &fileinfo)) != -1)
	{
		do
		{
			//检查是不是目录
			//如果不是,则进行处理
			if (!(fileinfo.attrib & _A_SUBDIR))
			{
				char filename[_MAX_PATH];
				strcpy(filename, dir);
				strcat(filename, fileinfo.name);
				cout << filename << endl;
				if (!ProcessFile(filename))
					return false;
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	//查找dir中的子目录
	//因为在处理dir中的文件时，派生类的ProcessFile有可能改变了
	//当前目录，因此还要重新设置当前目录为dir。
	//执行过_findfirst后，可能系统记录下了相关信息，因此改变目录
	//对_findnext没有影响。
	_chdir(dir);
	if ((hFile = _findfirst("*.*", &fileinfo)) != -1)
	{
		do
		{
			//检查是不是目录
			//如果是,再检查是不是 . 或 .. 
			//如果不是,进行迭代
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp
					(fileinfo.name, "..") != 0)
				{
					char subdir[_MAX_PATH];
					strcpy(subdir, dir);
					strcat(subdir, fileinfo.name);
					strcat(subdir, "\\");
					ProcessDir(subdir, dir);
					if (!BrowseDir(subdir, filespec))
						return false;
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return true;
}

vector<string> CBrowseDir::GetDirFilenames(const char *dir, const char *filespec)
{
	_chdir(dir);
	vector<string>filename_vector;
	filename_vector.clear();

	//首先查找dir中符合要求的文件
	long hFile;
	_finddata_t fileinfo;
	if ((hFile = _findfirst(filespec, &fileinfo)) != -1)
	{
		do
		{
			//检查是不是目录
			//如果不是,则进行处理
			if (!(fileinfo.attrib & _A_SUBDIR))
			{
				char filename[_MAX_PATH];
				strcpy(filename, dir);
				strcat(filename, fileinfo.name);
				filename_vector.push_back(filename);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	//查找dir中的子目录
	//因为在处理dir中的文件时，派生类的ProcessFile有可能改变了
	//当前目录，因此还要重新设置当前目录为dir。
	//执行过_findfirst后，可能系统记录下了相关信息，因此改变目录
	//对_findnext没有影响。
	_chdir(dir);
	if ((hFile = _findfirst("*.*", &fileinfo)) != -1)
	{
		do
		{
			//检查是不是目录
			//如果是,再检查是不是 . 或 .. 
			//如果不是,进行迭代
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp
					(fileinfo.name, "..") != 0)
				{
					char subdir[_MAX_PATH];
					strcpy(subdir, dir);
					strcat(subdir, fileinfo.name);
					strcat(subdir, "\\");
					ProcessDir(subdir, dir);
					vector<string>tmp = GetDirFilenames(subdir, filespec);
					for (vector<string>::iterator it = tmp.begin(); it<tmp.end(); it++)
					{
						filename_vector.push_back(*it);
					}
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return filename_vector;
}

bool CBrowseDir::ProcessFile(const char *filename)
{
	return true;
}

void CBrowseDir::ProcessDir(const char
	*currentdir, const char *parentdir)
{
}

//从CBrowseDir派生出的子类，用来统计目录中的文件及子目录个数
class CStatDir :public CBrowseDir
{
protected:
	int m_nFileCount;   //保存文件个数
	int m_nSubdirCount; //保存子目录个数

public:
	//缺省构造器
	CStatDir()
	{
		//初始化数据成员m_nFileCount和m_nSubdirCount
		m_nFileCount = m_nSubdirCount = 0;
	}

	//返回文件个数
	int GetFileCount()
	{
		return m_nFileCount;
	}

	//返回子目录个数
	int GetSubdirCount()
	{
		//因为进入初始目录时，也会调用函数ProcessDir，
		//所以减1后才是真正的子目录个数。
		return m_nSubdirCount - 1;
	}

protected:
	//覆写虚函数ProcessFile，每调用一次，文件个数加1
	virtual bool ProcessFile(const char *filename)
	{
		m_nFileCount++;
		return CBrowseDir::ProcessFile(filename);
	}

	//覆写虚函数ProcessDir，每调用一次，子目录个数加1
	virtual void ProcessDir
		(const char *currentdir, const char *parentdir)
	{
		m_nSubdirCount++;
		CBrowseDir::ProcessDir(currentdir, parentdir);
	}
};


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

		//构造类对象
		CStatDir statdir;

		//设置要遍历的目录
		if (!statdir.SetInitDir(image_filename))
		{
			puts("目录不存在。");
			return;
		}

			
			bool usesummary = atoi(limit) > 1;// strcmp(label_filename, "summary") == 0;
			std::ifstream image_file;
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
				rows1 = 19;
				cols1 = 38;
			}

			//title:
			bool isdebug = (limit_num == 9999);

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

		//开始遍历
			// Storing to db
			int label;
			unsigned char* tpixels = new unsigned char[2000];
			unsigned char* pixels = new unsigned char[rows1 * cols1];
			unsigned char win;
			int count = 0;
			const int kMaxKeyLength = 10;
			char key_cstr[kMaxKeyLength];
			string value;
			int mpos = 0;
			int mpos2 = 0;
			int label_i = 0;
			int spos = 0;
			int spos1 = 0;
			int empty_line = 0;
			//string keyword = "大家好,欢迎你";
			//strCoding cfm;


			Datum datum;
			datum.set_channels(1);
			datum.set_height(rows1);
			datum.set_width(cols1);
			//LOG(INFO) << "A total of " << num_items << " items.";
			LOG(INFO) << "Rows: " << rows1 << " Cols: " << cols1;
			string sline;
			char GOfile[5000];
			char xx, yy;
			int item_id = 0;
			//for (; item_id < start_num + num_items; ) {

		vector<string>file_vec = statdir.BeginBrowseFilenames("*.*");
		for (vector<string>::const_iterator it = file_vec.begin(); it < file_vec.end(); ++it)
		{
			LOG(INFO) << "open file :" << *it;
			std::ifstream image_file(*it, std::ios::in | std::ios::binary);


			if (usesummary){
				sline.clear();
				//getline(image_file, sline, '\n');
				memset(GOfile, 0, 5000);
				image_file.read(GOfile, 5000);
				sline = GOfile;
				int start_ = 0;
				mpos = sline.find("RE[");
				if (!(start_num == 1 || start_num == 2))
				{
					if (mpos > 1000 || mpos < 10 || sline.length() < 200 || sline.length() - mpos < 500){
						LOG(INFO) << "error file :mpos" << mpos << "sline.length " << sline.length();
						continue;
					}
					mpos2 = sline.find_first_of(']', mpos + 1);
					win = (GOfile[mpos + 3]);
				}
				else{
					mpos = 0;
				}

				if (win == 'W' || win == 231 || win == (unsigned char)0xB0 || start_num==1)//baiwin
				{
					win = 'W';
					//label_i = '0';
				}
				else if (win == 'B' || win == 233 || win == (unsigned char)0xBA || start_num==2)//heiwin
				{
					win = 'B';
					//label_i = '1';
				}
				else{
					LOG(INFO) << "win:ERROR : " << win << " mpos " << mpos ;
					continue;
				}
				if (isdebug)
					LOG(INFO) << "win:" << win;

				memset(tpixels, 0, 2000);
				memset(pixels, 0, rows1*cols1);

				//memcpy(tpixels, sline.substr(mpos2 + 1).c_str(), mpos);
				spos1 = 0;
				mpos2 = sline.find("B[", mpos + 1);
				int start_pos = mpos2;
				start_ = 0;

				while (mpos < sline.length() - 6 && item_id < start_num + num_items){

					spos = sline.find_first_of('(', mpos2 + 1);
					mpos = sline.find_first_of('[', mpos2 + 1);
					if (spos <mpos && spos>mpos2 && spos != -1)
					{
						LOG(INFO) << "found fenzi mpos [:" << mpos << " mpos 2 ]:" << mpos2 << " pos (:" << spos;
						spos1 = sline.find_first_of(')', spos + 1);
						if (spos1 - spos  > spos - start_pos && spos - start_pos < 40 * 6 && spos1 - spos >40 * 6)
						{
							
							//use new fenzi
							LOG(INFO) << "use new fenzi:len:" << spos1 - spos << " old len:" << spos - start_pos << " pos1 ):" << spos1;
							memset(pixels, 0, rows1*cols1);
							start_pos = spos;
							start_ = 0;
							mpos = sline.find("B[", spos + 1)+1;

						}
						else
							mpos = sline.find_first_of('[', spos1 + 1);
						
					}

					mpos2 = sline.find_first_of(']', mpos + 1);
					if (mpos2 - mpos != 3)
					{
						//if (isdebug)
						LOG(INFO) << "not found B[: mpos :" << mpos << " mpos 2:" << mpos2;
						break;
					}

					xx = GOfile[mpos + 1] - 'a';
					yy = GOfile[mpos + 2] - 'a';

					if (xx < 0 || xx>18 || yy < 0 || yy >18){
						LOG(INFO) << "not found xx yy";
						break;
					}


					
					if (GOfile[mpos -1] != win){

						pixels[  yy*cols1 + xx ] = 1;

						spos = sline.find_first_of("[", mpos2 + 1);
						spos1 = sline.find_first_of("]", spos + 1);
						if (!(start_num == 1 || start_num == 2)&&(spos1 - spos != 3 ))
						{
							//if (isdebug)
							LOG(INFO) << "not found next[: mpos2:<" << mpos2<<" pos :" << spos << " spos 2 : " << spos1;
							break;
						}
						if (start_num == 1 || start_num == 2)
						{
							if (spos1 - spos != 3)
							{
							}
							else{
								start_++;
								continue;
							}
						}

						xx = GOfile[spos + 1] - 'a';
						yy = GOfile[spos + 2] - 'a';

						if (start_ > 30 || (start_num == 1 || start_num == 2))
						{
							if ((start_num == 1 || start_num == 2))
							{
								xx = 10;
								yy = 10;
								memset(tpixels, 0, rows1*cols1);
								
								for (int ii = 0; ii < cols1/2; ii++)
									for (int jj = 0; jj < rows1; jj++)

										for (int ii2 = cols1 / 2; ii2 < cols1; ii2++)
											for (int jj2 = 0; jj2 < rows1; jj2++)
												if (pixels[jj*cols1 + ii] ==0 && pixels[jj2*cols1 + ii2 + cols1 / 2] ==0)
												{
													if (( ii2 % (cols1 / 2) == ii) && jj == jj2)
														continue;
													memcpy(tpixels, pixels, rows1*cols1);

													tpixels[jj*cols1 + ii] = 1;
													tpixels[jj2*cols1 + ii2 + cols1 / 2] = 1;
													label = jj*cols1 + ii;
													datum.set_data(tpixels, rows1*cols1);

									datum.set_label(label);
									sprintf_s(key_cstr, kMaxKeyLength, "%09d", item_id);
									item_id++;
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
							}
							if (isdebug)
								LOG(INFO) << "start write db";
							label = yy * 19 + xx;
							if (isdebug)
								LOG(INFO) << "label xx  " << GOfile[spos + 1] << " yy " << GOfile[spos + 2];

							datum.set_data(pixels, rows1*cols1);
							datum.set_label(label);
							sprintf_s(key_cstr, kMaxKeyLength, "%09d", item_id);
							item_id++;
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
					}
					else if (*sline.substr(mpos - 1, mpos).c_str() == win)
					{
						pixels[ yy*cols1 + xx + cols1/2] = 1;
					}

					//if (!start_)
						start_ ++;

				}

				if (isdebug)//99999 just for testing
				{
					memset(tpixels, 0, 2000);
					spos1 = 0;
					for (spos = 0; spos < rows1*cols1; spos++){
						if (pixels[spos] > 0){

							if (spos % (cols1 / 2) < cols1 / 2)//loss
							{

							}
							else
							{

							}
							tpixels[spos1++] = (uint8_t)((uint8_t)(spos % (cols1 / 2)) + 'a');//xx
							tpixels[spos1++] = (uint8_t)((uint8_t)(spos / cols1) + 'a');//yy
							
							tpixels[spos1++] = ';';

						}
					}
					tpixels[spos1++] = 0;
					LOG(INFO) << "spos1:" << spos1 << " tpixels:" << tpixels;
					//LOG(INFO) << " tpixels:" << (char *)(tpixels);
					//LOG(INFO) << " pixels:" << pixels;
				}


			}


		}

		//}
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
