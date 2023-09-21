# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/7/16 11:04 AM
@desc:
"""
from torch.utils.data import Dataset


class DynamicDataset(Dataset):
    """learning dynamics Dataset"""

    def __init__(self, dataset, transform=None):
        # 解析文件内容
        self.dataset = dataset
        self.transform = transform

        y0, y, yt, t, physics_t = self.dataset

        self.y0 = y0
        self.y = y
        self.yt = yt
        self.t = t
        self.physics_t = physics_t

    def __len__(self):
        # return len(self.physics_t)
        return 1

    def __getitem__(self, index):
        y0 = self.y0
        y = self.y
        yt = self.yt
        t = self.t
        physics_t = self.physics_t

        if self.transform is not None:
            y0 = self.transform(y0)
            y = self.transform(y)
            yt = self.transform(yt)
            t = self.transform(t).requires_grad_(True)
            physics_t = self.transform(physics_t).requires_grad_(True)

        return y0, y, yt, t, physics_t
