# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:18 PM
@desc:
"""
import os

import numpy as np


def read_data(data_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(data_path):
        raise IOError("{} does not exist".format(data_path))
    while not got_img:
        try:
            data = np.load(data_path, allow_pickle=True).item()
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(data_path))
            pass
    return data
