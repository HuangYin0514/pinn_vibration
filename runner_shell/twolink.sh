#!/bin/bash


################################
# dataset
################################
python ./task/generate.py --config_file configs/generate_data/two_link/config_generate_data.py

################################
# network
################################
python ./task/train.py --config_file configs/train/two_link/config_pinn.py
