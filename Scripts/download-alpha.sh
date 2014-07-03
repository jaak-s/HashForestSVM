#!/bin/bash

echo "Downloading alpha data (500k samples) into Data folder"

pushd "$(git rev-parse --show-toplevel)/Data" 
wget ftp://largescale.ml.tu-berlin.de/largescale/alpha/alpha_train.dat.bz2
wget ftp://largescale.ml.tu-berlin.de/largescale/alpha/alpha_train.lab.bz2
bunzip2 xf *.bz2
popd

