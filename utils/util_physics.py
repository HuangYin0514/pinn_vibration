# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/3/30 2:54 PM
@desc:
"""
import numpy as np
import torch


def dfx(f, x):
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]


def matrix_inv(b_mat):
    eye = torch.eye(b_mat.shape[1], dtype=b_mat.dtype, device=b_mat.device).expand_as(b_mat)
    b_inv = torch_linalg_solve_bug(b_mat, eye)
    return b_inv


def torch_linalg_solve_bug(L, R):
    """
    When dealing with batchsize=1, the results of cpu and gpu processing are inconsistent
    """
    bs, *star_L = L.shape
    _, *star_R = R.shape
    if L.shape[0] == 1:
        L_rpt = torch.cat([L, L], dim=0)
        R_rpt = torch.cat([R, R], dim=0)
        res = torch.linalg.solve(L_rpt, R_rpt)[0:1]
        # raise Exception('L batchsize is 1')
    else:
        res = torch.linalg.solve(L, R)
    return res


def ham_J(M):
    """
    applies the J matrix to another matrix M.
    input: M (*,2nd,bs)

    J ->  # [ 0, I]
          # [-I, 0]

    output: J@M (*,2nd,bs)
    """
    *star, D, b = M.shape
    JM = torch.cat([M[..., D // 2 :, :], -M[..., : D // 2, :]], dim=-2)
    return JM


def jacobian(y: torch.Tensor, x: torch.Tensor, need_higher_grad=True, device=None, dtype=None) -> torch.Tensor:
    # ref: https://zhuanlan.zhihu.com/p/530879775

    (Jac,) = torch.autograd.grad(
        outputs=(y.flatten(),),
        inputs=(x,),
        grad_outputs=(torch.eye(torch.numel(y), device=device, dtype=dtype),),
        create_graph=need_higher_grad,
        allow_unused=True,
        is_grads_batched=True,
    )
    if Jac is None:
        Jac = torch.zeros(size=(y.shape + x.shape), device=device, dtype=dtype)
    else:
        Jac.reshape(shape=(y.shape + x.shape))
    return Jac


def batched_jacobian(
    batched_y: torch.Tensor, batched_x: torch.Tensor, need_higher_grad=True, device=None, dtype=None
) -> torch.Tensor:
    #

    sumed_y = batched_y.sum(dim=0)  # y_shape
    J = jacobian(sumed_y, batched_x, need_higher_grad, device=device, dtype=dtype)  # y_shape x N x x_shape

    dims = list(range(J.dim()))
    dims[0], dims[sumed_y.dim()] = dims[sumed_y.dim()], dims[0]
    J = J.permute(dims=dims)  # N x y_shape x x_shape
    return J


def mat_vec_mul(mat, vec):
    """
    3D matrix-vector multiplication with batchsize,
    In GPUs, matmul is syntactic sugar for unknown results.

    Parameters
    ----------
    mat (bs, a, b)
    vec (bs, b)

    Returns
    mat (bs, a)
    -------

    """
    vec = vec.unsqueeze(-2)  # (bs, 1, b)
    vec = vec.repeat(1, mat.shape[1], 1)  # (bs, a, b)
    ele_mul = mat * vec  # (bs, a, b)
    res = torch.sum(ele_mul, dim=-1)  # (bs, a)
    return res
