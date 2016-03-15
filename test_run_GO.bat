rd /s /q Z:\GO-test-run-leveldb
#rd /s /q F:\caffe-windows\examples\GO\GO-test-run-leveldb

del  F:\caffe-windows\examples\GO\test-run\convert_GO_data.exe.DESKTOP-3POG0OV.teluw.log* /q

F:\caffe-windows\examples\GO\convert_GO_data.exe  F:\caffe-windows\examples\GO\test-run 1 9999 Z:\GO-test-run-leveldb --backend=leveldb

.\3rdparty\bin\caffe.exe test --debug=true --iterations=200 --weights=examples/GO/lenet_iter_24000.caffemodel --model=examples/GO/GO_demo_LeNet_test_run.prototxt

pause
