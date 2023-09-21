import os
import torch

####################################
# config.py

####################################
# For general settings
taskname = 'task_singlependlum_pinn'
seed = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.double  # torch.float32
outputs_dir = './outputs/'

####################################
# Dynamic
obj = 1
dim = 2
dof = obj * dim
m = (1., )
l = (1., )
g = 10

# ---------------------
dynamic_class = "DynamicSinglePendulumDAE"
# dynamic_class = "DynamicSinglePendulumDAE2ODE"

####################################
# data
data_name = 'DynamicData'
dataset_path = os.path.join(outputs_dir, "data", dynamic_class)
physic_num = 1000
interval = 10

####################################
# net
net_name = 'PINN'
load_net_path = ''

BackboneNet_input_dim = 1
BackboneNet_hidden_dim = 20
BackboneNet_output_dim = dof
BackboneNet_layers_num = 3

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
