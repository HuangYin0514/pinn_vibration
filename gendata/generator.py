import time
import numpy as np
import torch
from torch import nn
from integrator import ODEIntegrate
from utils import initialize_class, tensors_to_numpy


class DataGenerator(nn.Module):

    def __init__(self, config, logger):
        """
        数据生成器类，用于生成模拟数据并保存到文件。
        
        Args:
            config (dict): 配置参数
            logger: 日志记录器
        """
        super(DataGenerator, self).__init__()
        self.config = config
        self.logger = logger
        self._initialize_dynamics()

    def _initialize_dynamics(self):
        """
        初始化动力学模型。
        """
        try:
            class_name = self.config.dynamic_class
            self.logger.info("loading class: {}".format(class_name))
            kwargs = {"config": self.config, "logger": self.logger}
            self.right_term_net = initialize_class("dynamics", class_name, **kwargs)
        except ValueError as e:
            raise RuntimeError(f"Class '{class_name}' is not available")

    def _generate_initial_configs(self):
        """
        生成随机的初始配置。
        """
        y0 = torch.tensor(self.config.y0, device=self.config.device, dtype=self.config.dtype)
        y0 = y0.repeat(self.config.data_num, 1)
        return y0

    def _integrate_ode(self, y0):
        """
        解ODE方程, 获取模拟数据。
        
        Args:
            y0: 初始配置
            
        Returns:
            tuple: (时间数组, 解数组)
        """
        start_time = time.time()

        t, sol = ODEIntegrate(func = self.right_term_net,
                              t0 = self.config.t0,
                              t1 = self.config.t1,
                              dt = self.config.dt,
                              y0 = y0,
                              method=self.config.ode_solver,
                              device=self.config.device,
                              dtype=self.config.dtype,
                              dof=self.config.dof)

        end_time = time.time()
        execution_time = end_time - start_time
        self.logger.info(f"The running time of ODE solver: {execution_time} s")

        return t, sol

    def _truncated_lambdas(self, values):
        """
        截断lambda值。
        
        Args:
            values: lambda值
            
        Returns:
            torch.Tensor: 截断后的lambda值
        """
        q, qt, lambdas = torch.tensor_split(values, (self.config.dof, self.config.dof * 2), dim=-1)
        return torch.cat([q, qt], dim=-1)

    def generate_data(self, dataset_name, *args, **kwargs):
        """
        生成模拟数据并保存到文件。
        
        Args:
            dataset_name (str): 数据集文件名
        """
        y0 = self._generate_initial_configs(*args, **kwargs)
        t, y = self._integrate_ode(y0=y0, *args, **kwargs)
        yt = torch.stack([self.right_term_net(t, yi).clone().detach().cpu() for yi in y])
        # yt = torch.stack([self.right_term_net(t, yi.view(1,-1)).clone().detach().cpu() for yi in y[0]])

        # 仅取一条轨迹数据
        y0 = y0[0]
        y = y[0]
        yt = yt.squeeze()

        # 截断lambda值
        y0 = self._truncated_lambdas(y0)
        y = self._truncated_lambdas(y)
        yt = self._truncated_lambdas(yt)

        # to numpy
        y0, t, y, yt = tensors_to_numpy(y0, t, y, yt)

        # 提取一个初始值
        dataset = {
            'y0': y0,
            't': t,
            'y': y,
            'yt': yt,
            't0': self.config.t0,
            't1': self.config.t1,
            'dt': self.config.dt,
        }

        y0_output = "y0: {}".format(y0)
        t_output = "t: {}".format(t)
        y_output = "yt: {}".format(y)
        yt_output = "yt: {}".format(yt)
        t0_output = "t0: {}".format(self.config.t0)
        t1_output = "t1: {}".format(self.config.t1)
        dt_output = "dt: {}".format(self.config.dt)
        full_output = "\n----------\n ".join([y0_output, t_output, y_output, yt_output, t0_output, t1_output, dt_output])

        # Log the dataset values
        self.logger.debug("dataset values:\n%s", full_output)

        np.save(dataset_name, dataset)
