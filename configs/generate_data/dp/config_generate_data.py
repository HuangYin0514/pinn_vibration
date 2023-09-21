import os
import torch

####################################
# config.py
####################################
# For general settings
taskname = 'generate_double_pendulum_data'
seed = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.double  # torch.float32
outputs_dir = './outputs/'

####################################
# Dynamic
obj = 2
dim = 2
dof = obj * dim
m = (1., 1.)
l = (1., 1.)
g = 10

# ---------------------
dynamic_class = "DynamicDoublePendulumDAE"
lambda_len = 2
# dynamic_class = "DynamicDoublePendulumDAE2ODE"
# lambda_len = 0

q = [1, 0, 2, 0]
qt = [0] * 4
lam = [0] * lambda_len
y0 = q + qt + lam

####################################
# For Solver settings
t0 = 0.0
t1 = 5.0
dt = 0.01
data_num = 1
ode_solver = 'RK4'
# ode_solver = 'Euler'
# ode_solver = 'ImplicitEuler'

####################################
# For outputs settings
dataset_path = os.path.join(outputs_dir, "data", dynamic_class)
