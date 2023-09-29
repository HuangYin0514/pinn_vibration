import os
import torch

####################################
# config.py

####################################
# For general settings
taskname = 'task_burgers_equation'
seed = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # torch.float32 / torch.double
outputs_dir = './outputs/'

####################################
# data
data_name = 'DataBurgers'
dataset_path = "/home/lbu/project/pinn_vibration/learner/data/datasets/burgers_equation/Data/burgers_shock.mat"

