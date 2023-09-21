#!/bin/bash


################################
# dataset
################################
python ./task/generate.py --config_file configs/generate_data/slider_crank/config_generate_data.py
# python ./task/generate.py --config_file configs/generate_data/slider_crank/config_generate_data_h001_t1.py

################################
# network
################################
python ./task/train.py --config_file configs/train/slider_crank/config_pinn.py
# python ./task/train.py --config_file configs/train/slider_crank/config_pirnn.py
# python ./task/train.py --config_file configs/train/slider_crank/config_pinn_two_input_port.py
