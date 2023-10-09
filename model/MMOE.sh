#!/bin/bash
source /etc/profile
set -e
# here is actually yesterday
today=$(date "+%Y%m%d" -d "-1day")
/root/miniconda3/envs/py368_tf115/bin/python3.6 /git/alg_rep/MODELS/HUDONG/CVR/CMoE/model/MMOE.py \
--task_type=train \
--learning_rate=0.0006 \
--optimizer=Adam \
--num_epochs=1 \
--num_experts=20 \
--num_tasks=3 \
--batch_size=10240 \
--field_size=75 \
--feature_size=1500000 \
--deep_layers=512,256,128 \
--dropout=0.9,0.9,0.9 \
--log_steps=30 \
--num_threads=12 \
--clear_existing_model=True

/root/miniconda3/envs/py368_tf115/bin/python3.6 /git/alg_rep/MODELS/HUDONG/CVR/CMoE/model/MMOE.py \
--task_type=eval \
--learning_rate=0.0006 \
--optimizer=Adam \
--num_epochs=1 \
--num_experts=20 \
--num_tasks=3 \
--batch_size=10240 \
--field_size=75 \
--feature_size=1500000 \
--deep_layers=512,256,128 \
--dropout=0.9,0.9,0.9 \
--log_steps=30 \
--num_threads=12 \

#servable_model_dir=$"/data/servable_model/""servable_model_"$today"_mmoe"
#/root/miniconda3/envs/py368_tf115/bin/python3.6 /git/alg_rep/MODELS/HUDONG/CVR/MMOE2.py \
#--task_type=export \
#--learning_rate=0.0006 \
#--optimizer=Adam \
#--num_epochs=1 \
#--num_experts=20 \
#--num_tasks=3 \
#--batch_size=10240 \
#--field_size=75 \
#--feature_size=1500000 \
#--deep_layers=512,256,128 \
#--dropout=0.9,0.9,0.9 \
#--log_steps=30 \
#--num_threads=12 \
#--servable_model_dir=$servable_model_dir
