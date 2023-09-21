import os
import numpy as np
import torch

####################################
# config.py
####################################
# For general settings
taskname = "generate_scissor_space_deployable_data"
seed = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.double  # torch.float32
outputs_dir = "./outputs/"

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
dynamic_class = "DynamicScissorSpaceDeployableDAE"
lambda_len = 10 + 19 * unitnum  # 105

height = (l[0] ** 2 - l[1] ** 2) ** 0.5  # 折叠状态单个剪叉单元高度
q = np.zeros(4 * 3 * (unitnum + 1))
for i in range(unitnum + 1):
    q[12 * i : 12 * i + 12] = np.array(
        [
            l[1] / 2,
            -l[1] / 2,
            i * height,
            l[1] / 2,
            l[1] / 2,
            i * height,
            -l[1] / 2,
            l[1] / 2,
            i * height,
            -l[1] / 2,
            -l[1] / 2,
            i * height,
        ]
    )
q = q.tolist()
qt = [0] * len(q)
lam = [0] * lambda_len
y0 = q + qt + lam

####################################
# For Solver settings
t0 = 0.0
t1 = 10.0
dt = 0.01
data_num = 1
ode_solver = "RK4_high_order"
# ode_solver = "RK4"
# ode_solver = 'Euler'
# ode_solver = 'ImplicitEuler'

####################################
# For outputs settings
dataset_path = os.path.join(outputs_dir, "data", dynamic_class)
