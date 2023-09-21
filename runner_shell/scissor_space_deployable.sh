#!/bin/bash


################################
# dataset
################################
python ./task/generate.py --config_file configs/generate_data/scissor_space_deployable/config_generate_data.py

################################
# network
################################
python ./task/train.py --config_file configs/train/scissor_space_deployable/config_pinn.py
