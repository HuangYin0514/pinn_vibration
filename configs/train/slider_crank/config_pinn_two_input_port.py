import os
import torch
import numpy as np

####################################
# config.py

####################################
# For general settings
taskname = "task_slider_crank_pinn_two_input_port"
seed = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.double  # torch.float32
outputs_dir = "./outputs/"

####################################
# Dynamic
obj = 3
dim = 3
dof = obj * dim - 2
m = (1.0, 1.0, 2.0)
l = (1.0, np.sqrt(3))
g = 10

# ---------------------
# dae
dynamic_class = "DynamicSliderCrankDAE"

####################################
# data
data_name = "DynamicData"
dataset_path = os.path.join(outputs_dir, "data", dynamic_class)
interval = 1

####################################
net_name = "PINN_two_input_port"
load_net_path = ""

BackboneNet_input_dim = 1 + 2 * dof
BackboneNet_hidden_dim = 20
BackboneNet_output_dim = dof
BackboneNet_layers_num = 3

loss_y0_weight = 30
loss_data_weight = 1
loss_physic_weight = 0.1

####################################
# For training settings
learning_rate = 1e-3
optimizer = "adam_LBFGS"
scheduler = "no_scheduler"
iterations = 1500
optimize_next_iterations = 1400
print_every = 100
