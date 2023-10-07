# 求解 1D lateral vibration 方程反问题


## 引言

- 方程信息
  
  $$ 
  \begin{aligned}

      & \frac{\partial^{2}u}{\partial t^{2}}=-\alpha^{2}\frac{\partial^{4}u}{\partial x^{4}} ,\quad x\in[0,1],\quad t\in[0,1],  \\

      & u(0,x)=\sin(\pi x), \\

      & u(t,0)=u(t,1)=0.

  \end{aligned}
  $$
  其中，$\alpha=1.0$。

- 方程解图像
  
  ![Alt text](figures/equation_physics.jpg)
  ![Alt text](figures/equation_solution.jpg)

- 解析解

  $$ 
  \begin{aligned}

      & u = \text{sin}(\pi x) \text{cos}(\pi^2 t)

  \end{aligned}
  $$

## 方法

- PINN求解 1D lateral vibration 方程

- 训练文件[shell](run.sh)
    
## 结果

- 结果文件[infer](analysis/infer.ipynb)
- 结果

    Train completion time: 2023-10-05-17-17-32
    Task name: task_1D_longitudinal_vibration
    Model name: PINN
    Best model at iteration: 4036
    Train loss: 1.895e-05
    Val loss: 6.050e-03
    
    ![Alt text](figures/loss_curve.png)
    ![Alt text](figures/1D_longitudinal_vibration_result.png)


## 总结

- None