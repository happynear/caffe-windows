..\..\..\3rdparty\bin\protoc caffe.proto --cpp_out=./
..\..\..\3rdparty\bin\protoc caffe.proto --python_out=./
md ..\..\..\python\caffe\proto\
copy /y .\caffe_pb2.py ..\..\..\python\caffe\proto\
copy nul ..\..\..\python\caffe\proto\__init__.py
pause