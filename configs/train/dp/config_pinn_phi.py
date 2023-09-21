import os
import torch

####################################
# config.py

####################################
# For general settings
taskname = 'task_doublependlum_pinn_phi'
seed = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.double  # torch.float32
outputs_dir = './outputs/'

####################################
# Dynamic
obj = 2
dim = 2
dof = obj * dim
m = (500., 1.)
l = (1., 1.)
g = 10

# ---------------------
# dae
dynamic_class = "DynamicDoublePendulumDAE"
# dynamic_class = "DynamicDoublePendulumDAE2ODE"

####################################
# data
data_name = 'DynamicData'
dataset_path = os.path.join(outputs_dir, "data", dynamic_class)
physic_num = 1000
interval = 10

####################################
# net
net_name = 'PINN_phi'
load_net_path = ''

BackboneNet_input_dim = 1
BackboneNet_hidden_dim = 20
BackboneNet_output_dim = dof
BackboneNet_layers_num = 3

loss_y0_weight = 1
loss_data_weight = 1
loss_physic_weight = 1

####################################
# For training settings
learning_rate = 1e-3
optimizer = 'adam_LBFGS'
scheduler = 'no_scheduler'
iterations = 1500
optimize_next_iterations = 1400
print_every = 100
