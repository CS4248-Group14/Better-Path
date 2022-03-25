#!/bin/bash

if [ ! -d split_feature_chunks ]; then
  mkdir split_feature_chunks
  cd split_feature_chunks
  wget https://github.com/YilunZhou/path-naturalness-prediction/raw/master/code/science/split_feature_chunks/xa
  wget https://github.com/YilunZhou/path-naturalness-prediction/raw/master/code/science/split_feature_chunks/xb
  wget https://github.com/YilunZhou/path-naturalness-prediction/raw/master/code/science/split_feature_chunks/xc
  cd ..
fi

