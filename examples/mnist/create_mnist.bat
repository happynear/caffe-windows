rd /s /q mnist_train_lmdb 
rd /s /q mnist_test_lmdb 
convert_mnist_data.exe data/train-images-idx3-ubyte data/train-labels-idx1-ubyte mnist_train_lmdb --backend=lmdb
convert_mnist_data.exe data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte mnist_test_lmdb --backend=lmdb
pause