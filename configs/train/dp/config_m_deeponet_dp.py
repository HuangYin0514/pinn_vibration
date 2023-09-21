import os
import torch

####################################
# config.py

####################################
# For general settings
taskname = 'task_doublependlum_pinn_m_deeponet'
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
net_name = 'M_DeepONet'
load_net_path = ''

ScaleNet_input_dim = 1
ScaleNet_hidden_dim = 4
ScaleNet_blocks_num = 3
ScaleNet_output_dim = ScaleNet_blocks_num * ScaleNet_hidden_dim + ScaleNet_input_dim

TrunkNet_input_dim = ScaleNet_output_dim
TrunkNet_hidden_dim = 20
TrunkNet_output_dim = 5
TrunkNet_layers_num = 3

BranchNet_input_dim = 8
BranchNet_hidden_dim = 20
BranchNet_output_dim = 5
BranchNet_layers_num = 1

dynamic_layer_input_dim = 10
dynamic_layer_output_dim = 4

loss_y0_weight = 30
loss_data_weight = 1
loss_physic_weight = 0.1

####################################
# For training settings
learning_rate = 1e-3
optimizer = 'adam_LBFGS'
scheduler = 'no_scheduler'
iterations = 1500
optimize_next_iterations = 1400
print_every = 100
