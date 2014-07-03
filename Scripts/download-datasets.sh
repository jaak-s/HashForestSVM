#!/bin/bash

gitdir="$(git rev-parse --show-toplevel)"

if [ -z "$gitdir" ]; then 
  echo "Please run the script inside HashForestSVM git directory."
  exit 1
fi

if [ ! -d "$gitdir/Data" ]; then
  mkdir "$gitdir/Data"
fi

pushd $gitdir/Data

## CIFAR: http://www.cs.toronto.edu/~kriz/cifar.html
echo "Downloading CIFAR-10."
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# Potential data sources:
# 1) MED11
# 2) CIFAR10
# 3) IJCNN
# 4) WebSpam
# 5) ImageNet

popd

