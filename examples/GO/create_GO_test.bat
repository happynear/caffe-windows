
rd /s /q F:\caffe-windows\examples\GO\GO-test-leveldb

del .\GOqiku\convert_GO_data.exe.DESKTOP-3POG0OV.teluw.log* /q
convert_GO_data.exe .\GOqiku 0 10000 F:\caffe-windows\examples\GO\GO-test-leveldb --backend=leveldb

pause