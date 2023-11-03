# 求解 Burgers 方程


## 引言

- Burgers 方程信息
  
  $$ 
  \begin{aligned}
    &u_{t}+uu_{x}-(0.01/\pi)u_{xx}=0,\quad x\in[-1,1],\quad t\in[0,1], \\
    &u(0,x)=-\sin(\pi x), \\
    &u(t,-1)=u(t,1)=0.
  \end{aligned}
  $$

- Burgers 方程图像
  ![burgers equation](figures/Burgers_CT_inference.png)

## 方法

- PINN求解 Burgers 方程

- 训练文件[shell](run.sh)
    
## 结果

- 结果文件[infer](analysis/infer.ipynb)
- 结果

    Train completion time: 2023-10-05-14-18-08
    Task name: burgers_task
    Model name: PINN
    Best model at iteration: 4246
    Train loss: 5.829e-06
    Val loss: 5.138e-06

    ![loss_curve](figures/loss_curve.png)
    ![Burgers_results](figures/Burgers_results.png)

## 总结

- None