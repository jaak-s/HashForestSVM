HashForestSVM
=============

Hash-based trees for speeding up Support Vector Machines.

# Installation
Install external packages:
```bash
sudo pip3 install scikit
sudo easy_install3 cython
```

Then clone git and compile cython code
```bash
git clone git@github.com:jaak-s/HashForestSVM.git
cd HashForestSVM/HashForestSVM/cycode
python3 setup.py build_ext --inplace
```

# Downloading data sources
```bash
Scripts/download-datasets.sh
```

