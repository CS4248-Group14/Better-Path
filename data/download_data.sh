#!/bin/bash

for dir in science money
do
  if [ ! -d $dir ]; then
    echo download $dir
    mkdir $dir
    cd $dir
    wget https://raw.githubusercontent.com/YilunZhou/path-naturalness-prediction/master/data/$dir/paths.pkl
    wget https://raw.githubusercontent.com/YilunZhou/path-naturalness-prediction/master/data/$dir/answers.txt
    cd ..
  fi
done

if [ ! -d open-domain ]; then
  echo open-domain
  mkdir open-domain
  cd open-domain
  wget https://raw.githubusercontent.com/YilunZhou/path-naturalness-prediction/master/data/open-domain/answers.txt
  wget https://raw.githubusercontent.com/YilunZhou/path-naturalness-prediction/master/data/open-domain/paths-len2.pkl
  wget https://raw.githubusercontent.com/YilunZhou/path-naturalness-prediction/master/data/open-domain/paths-len3.pkl
  wget https://raw.githubusercontent.com/YilunZhou/path-naturalness-prediction/master/data/open-domain/paths-len4.pkl
  wget https://raw.githubusercontent.com/YilunZhou/path-naturalness-prediction/master/data/open-domain/paths-len5.pkl
  cd ..
fi

