#!/bin/bash


################################
# dataset
################################
python ./task/generate.py --config_file configs/generate_data/dp/config_generate_data.py
################################
# network
################################
python ./task/train.py --config_file configs/train/dp/config_pinn.py

