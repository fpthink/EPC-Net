#!/bin/sh
export PYTHONPATH=./

PYTHON=python
teacher_ckpt=$1
exp_name_stu=$2
config=$3

exp_dir_stu=exp/${exp_name_stu}

model_log_stu=${exp_dir_stu}/log
mkdir -p ${model_log_stu}

model_path_stu=${exp_dir_stu}/saved_model
mkdir -p ${model_path_stu}

model_events_stu=${exp_dir_stu}/events
mkdir -p ${model_events_stu}


now=$(date +"%Y%m%d_%H%M%S")

cp sh_kd_train.sh kd_train.py ${config} ${exp_dir_stu}

$PYTHON kd_train.py \
    --config ${config} \
    --save_path_student=${exp_dir_stu} \
    --resume_teacher=${teacher_ckpt} 2>&1 | tee ${model_log_stu}/train-$now.log

