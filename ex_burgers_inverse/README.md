# 求解 Burgers 方程反问题


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

    Train completion time: 2023-10-06-13-31-59
    Task name: burgers_inverse_task
    Model name: PINN
    Best model at iteration: 4239
    Train loss: 1.178e-04
    Val loss: 8.856e-07

    ![loss_curve](figures/loss_curve.png)
    ![Burgers_results](figures/Burgers_inverse_results.png)

    2023-10-06 14:26:08 INFO pred nu is 0.003180548781529069

    2023-10-06 14:26:08 INFO True nu is 0.003183098861837907

## 总结

- 提高数据损失的权重，数据权重30，方程权重1。