
rd /s /q F:\caffe-windows\examples\GO\GO-train-leveldb

del .\GOqiku\convert_GO_data.exe.DESKTOP-3POG0OV.teluw.log* /q
convert_GO_data.exe .\GOqiku 0 99999999 F:\caffe-windows\examples\GO\GO-train-leveldb --backend=leveldb

pause