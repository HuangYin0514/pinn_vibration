# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:20 PM
@desc:
"""
import torch


def train_collate_fn(batch):
    y0, y, yt, data_t, physics_t  = zip(*batch)

    y0 = y0[0]
    y = y[0]
    yt = yt[0]
    data_t = data_t[0]
    physics_t = physics_t[0]

    return y0, y, yt, data_t, physics_t


def val_collate_fn(batch):
    y0, y, yt, data_t, physics_t  = zip(*batch)

    y0 = y0[0]
    y = y[0]
    yt = yt[0]
    data_t = data_t[0]
    physics_t = physics_t[0]

    return y0, y, yt, data_t, physics_t
