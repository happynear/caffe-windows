**News:** Now the offical caffe has merged the windows fork maintained by [Microsoft](https://github.com/microsoft/caffe): https://github.com/BVLC/caffe/tree/windows . 

I think it is time to turn your code to the official version. This repository, `caffe-windows`, will be used for my personal research from right now. Thank you all guys who have been making contribution to the windows version of Caffe! 

People who do research in Facial Analysis or Visualization may continue to follow my `ms` branch([link](https://github.com/happynear/caffe-windows/tree/ms)). I will keep updating layers and scripts of these two areas. 

I have made a list of some frequently asked questions in [FAQ.md](FAQ.md). If you get confused during configuring, please firstly look up for your question in the [FAQ.md](FAQ.md). This FAQ list is still under construction, I will keep adding questions into it.

## Setup step:

1. Download third-party libraries from [BaiduYun Disk](http://pan.baidu.com/s/1eStyfrc) or [OneDrive](https://onedrive.live.com/redir?resid=A7AB61436C858EB9!1145&authkey=!AHlCN80o3Rx3n3E&ithint=file%2crar) and extract the files to `caffe-windows_root/3rdparty/`. **Please don't forget to add the `./3rdparty/bin` folder to your environment variable `PATH`.**

2. Run `./src/caffe/proto/extract_proto.bat` to create `caffe.pb.h`, `caffe.pb.cc` and `caffe_pb2.py`.

3. Double click ./buildVS2013/MainBuilder.sln to open the solution. If you do not have a Nvidia GPU, please open ./build_cpu_only/MainBuilder.sln.

4. Change the compile mode to Release and X64.

5. Modify the cuda device compute capability defined in the settings (`caffelib properties` -> `CUDA C/C++` -> `Device` -> `Code Generation`) to your GPU's compute capability (such as compute_30,sm_30; etc). You can look up for your GPU's compute capability in https://en.wikipedia.org/wiki/CUDA . Some general GPUs' compute capabilities are listed below.

 - If your GPU's compute capability is below or equal to 2.1, please remove the `USE_CUDNN` macro in the proprocessor definition of all projects.
 - If you are using cpu only solution, just ignore this step.

6. Compile.

| GPU                                         | Compute Capability    |
| ------------------------------------------- |:---------------------:|
| GTX660, 680, 760, 770                       | compute_30,sm_30      |
| GTX780, Titan Z, Titan Black, K20, K40      | compute_35,sm_35      |
| GTX960, 980, Titan X                        | compute_52,sm_52      |


中文安装说明：http://blog.csdn.net/happynear/article/details/45372231

### Matlab Wrapper

Just replace the Matlab include and library path defined in the settings and compile.
**Don't forget to add `./matlab` to your Matlab path.**

### Python Wrapper

Similar with Matlab, replace the python include and library path and compile.

Most of the libraries listed in `./python/requirements.txt` can be installed by `pip install`. However, some of them cannot be installed so easily.

For protobuf, you may download the codes from https://github.com/google/protobuf. Copy `caffe-windows-root/src/caffe/proto/protoc.exe` to `protobuf-root/src`. Then run `python setup.py install` in `protobuf-root/python`.

For leveldb, I have created a repository https://github.com/happynear/py-leveldb-windows . Please follow the instructions in `README.md` to install it.

### MNIST example

Please download the mnist leveldb database from https://pan.baidu.com/s/1mgl9ndu and extract it to `./examples/mnist`. Then double click `./run_mnist.bat` to run the MNIST demo.

## Update log

2015/11/09 CuDNN v3 works well now.

2015/09/14 Multi-GPU is supported now. 

WARNING: When you are using multiple gpus to train a model, please do not directly close the command window. Instead, please use `Ctrl+C` to avoid the gpu driver from crash.

You can also press `Ctrl+Break` to save a model snapshot whenever you want during training.

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
