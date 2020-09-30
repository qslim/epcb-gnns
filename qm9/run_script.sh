#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=9
export PYTHONIOENCODING=utf-8

dataset="QM9"
output_dir="../../epcb_results/"
config_file="./"$dataset".json"

time_stamp=`date '+%s'`
commit_id=`git rev-parse HEAD`
std_file=${output_dir}${time_stamp}_${commit_id}".txt"

nohup python -u ./main.py --config=$config_file --id=$commit_id --ts=$time_stamp --dir=$output_dir >> $std_file 2>&1 &
pid=$!

echo "Stdout dir:   $std_file"
echo "Start time:   `date -d @$time_stamp  '+%Y-%m-%d %H:%M:%S'`"
echo "CUDA DEVICES: $CUDA_VISIBLE_DEVICES"
echo "pid:          $pid"
cat $config_file

tailf $std_file
