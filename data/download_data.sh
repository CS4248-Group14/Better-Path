#!/bin/bash

for dir in science money open-domain
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

