import os
import numpy as np
import torch

####################################
# config.py
####################################
# For general settings
taskname = "generate_TwoLink_data"
seed = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.double  # torch.float32
outputs_dir = "./outputs/"

####################################
# Dynamic
obj = 2
dim = 3
dof = obj * dim
m = (1.0, 2.0)
l = (1.0, np.sqrt(3))
g = 10

# ---------------------
dynamic_class = "DynamicTwoLinkDAE"
lambda_len = 4

# theta1 = np.pi / 3
# theta2 = -np.pi / 6
theta1 = 0
theta2 = 0

q = [
    l[0] / 2 * np.cos(theta1),
    l[0] / 2 * np.sin(theta1),
    theta1,
    l[0] * np.cos(theta1) + l[1] / 2 * np.cos(theta2),
    l[0] * np.sin(theta1) + l[1] / 2 * np.sin(theta2),
    theta2,
]
qt = [0] * 6
lam = [0] * lambda_len
y0 = q + qt + lam

####################################
# For Solver settings
t0 = 0.0
t1 = 5.0
dt = 0.01
data_num = 1
ode_solver = "RK4"
# ode_solver = 'Euler'
# ode_solver = 'ImplicitEuler'

####################################
# For outputs settings
dataset_path = os.path.join(outputs_dir, "data", dynamic_class)
