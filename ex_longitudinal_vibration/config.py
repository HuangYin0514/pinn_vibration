import torch
import os

########################################################################
# config.py
########################################################################
# For general settings
taskname = "single_pendulum_task"
seed = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # torch.float32 / torch.double

########################################################################
# Dynamic
obj = 1
dim = 2
dof = obj * dim
m = (1.0,)
l = (1.0,)
g = 10

dynamic_class = "DynamicSinglePendulumDAE"
lambda_len = 1

q = [1, 0]
qt = [0] * 2
lam = [0] * lambda_len
y0 = q + qt + lam

########################################################################
# For Solver settings
t0 = 0.0
t1 = 5.0
dt = 0.01
data_num = 1
ode_solver = "RK4"
# ode_solver = 'Euler'
# ode_solver = 'ImplicitEuler'

########################################################################
# For training settings
learning_rate = 1e-3
optimizer = "adam"
scheduler = "no_scheduler"
iterations = 5000
optimize_next_iterations = 1400
print_every = 1000

########################################################################
# net
BackboneNet_input_dim = 1
BackboneNet_hidden_dim = 20
BackboneNet_output_dim = dof
BackboneNet_layers_num = 3


########################################################################
# For outputs settings
current_directory = os.path.dirname(os.path.realpath(__file__))
outputs_dir = "./outputs/"
dataset_path = os.path.join(current_directory, "dataset")
outputs_path = os.path.join(current_directory, outputs_dir)
