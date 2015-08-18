#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.

EXAMPLE=examples/mnist
DATA=data
BUILD=build/examples/mnist

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf mnist_train_${BACKEND}
rm -rf mnist_test_${BACKEND}

convert_mnist_data.exe $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte mnist_train_${BACKEND} --backend=${BACKEND}
convert_mnist_data.exe $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte mnist_test_${BACKEND} --backend=${BACKEND}

echo "Done."

pause