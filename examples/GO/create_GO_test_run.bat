
rd /s /q F:\caffe-windows\examples\GO\GO-test-run-leveldb

del .\test-run\convert_GO_data.exe.DESKTOP-3POG0OV.teluw.log* /q
convert_GO_data.exe .\test-run 0 362 F:\caffe-windows\examples\GO\GO-test-run-leveldb --backend=leveldb
