- [摘要](#摘要)
  - [实验1：ex\_burgers](#实验1ex_burgers)
  - [实验2：ex\_longitudinal\_vibration](#实验2ex_longitudinal_vibration)


# 摘要

利用神经网络求解PDE


## 实验1：ex_burgers
- PINN求解burgers方程
- burgers方程信息
  $$ 
  \begin{aligned}
  &u_{t}+uu_{x}-(0.01/\pi)u_{xx}=0,\quad x\in[-1,1],\quad t\in[0,1], \\
  &u(0,x)=-\sin(\pi x), \\
  &\begin{aligned}u(t,-1)=u(t,1)=0.\end{aligned}
  \end{aligned}
  $$
- burgers方程图像
  ![burgers equation](ex_burgers/figures/Burgers_CT_inference.png)

- 实验相关信息:[README.md](ex_burgers/README.md)
  
  
## 实验2：ex_longitudinal_vibration
- PINN求解Longitudinal vibration方程
- Longitudinal vibration方程信息
  $$\frac{\partial^2u}{\partial t^2}=\alpha^2\frac{\partial^2u}{\partial x^2}$$

- Longitudinal vibration方程图像
  ![方程图像](ex_longitudinal_vibration/figures/Longitudinal_vibration_equation.jpg)

- 实验相关信息:[README.md](ex_longitudinal_vibration/README.md)
  