import os
import torch
import numpy as np

####################################
# config.py

####################################
# For general settings
taskname = 'task_scissor_space_deployable_pinn'
seed = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.double  # torch.float32
outputs_dir = './outputs/'

####################################
# Dynamic
obj = 24
dim = 3
dof = obj * dim

unitnum = 5
l = (1.6, 1.59)

M_A = 0.03 * 0.03
M_rho = 3000

F_a = 2**0.5 / 2
F_f = 600  # 驱动力
F_r = 5  # 伸展臂载荷

# ---------------------
# dae
dynamic_class = "DynamicScissorSpaceDeployableDAE"

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
BackboneNet_hidden_dim = 50
BackboneNet_output_dim = dof
BackboneNet_layers_num = 3

loss_y0_weight = 30
loss_data_weight = 10
loss_physic_weight = 0.01

####################################
# For training settings
learning_rate = 1e-3
optimizer = 'adam_LBFGS'
scheduler = 'no_scheduler'
iterations = 1500
optimize_next_iterations = 1400
print_every = 500
