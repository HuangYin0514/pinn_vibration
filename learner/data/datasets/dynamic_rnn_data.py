import glob
import os
import numpy as np

from .utils import read_data


def parse_data_file(file_path):
    """
    解析数据文件，提取数据并返回。

    Parameters:
        file_path (str): 数据文件的路径。

    Returns:
        tuple: 包含 y0, y, dy, t, t0, t1, dt 的元组。
    """
    data = read_data(file_path)
    y0 = data["y0"]
    y = data["y"]
    yt = data["yt"]
    t = data["t"]
    t0 = data["t0"]
    t1 = data["t1"]
    dt = data["dt"]
    return y0, y, yt, t, t0, t1, dt


class DynamicRnnData:
    """
    动态数据加载器，用于从数据文件读取数据并准备训练和验证数据。

    Attributes:
        config (object): 包含数据加载器配置的对象。
        logger: (object): 日志记录器对象。
        data (tuple): 包含训练数据和验证数据的元组。
    """

    def __init__(self, config, logger):
        """
        初始化 DynamicRnnData 对象。

        Parameters:
            config (object): 包含数据加载器配置的对象。
            logger (object): 日志记录器对象。
        """
        self.config = config
        self.logger = logger

        y0, y, yt, t, t0, t1, dt = self._read_data(self.config.dataset_path)

        (
            train_y0,
            train_y,
            train_yt,
            train_t,
            train_physics_t,
        ) = self._prepare_train_data(y0, y, yt, t, t0, t1, dt)

        val_y0, val_y, val_yt, val_t, val_physics_t = self._prepare_val_data(
            y0, y, yt, t, t0, t1, dt
        )

        # 准备训练和验证数据
        train_data = train_y0, train_y, train_yt, train_t, train_physics_t
        val_data = val_y0, val_y, val_yt, val_t, val_physics_t

        self.data = train_data, val_data

    def _read_data(self, dataset_path):
        """
        从数据集目录读取数据文件并解析数据。

        Returns:
            tuple: 包含 y0, y, dy, t, t0, t1, dt 的元组。
        """
        if not dataset_path:
            raise ValueError("dataset_path is not provided")

        if not os.path.exists(dataset_path):
            raise ValueError("'{}' is not available".format(dataset_path))

        # 从数据集目录中读取数据文件
        file_path = glob.glob(os.path.join(dataset_path, "*.npy"))[0]
        y0, y, dy, t, t0, t1, dt = parse_data_file(file_path)
        self.logger.info("{} is loaded".format(self.__class__.__name__))

        # 重新格式化数据
        t = t.reshape(-1, 1)
        y0 = y0.reshape(1, -1)

        return y0, y, dy, t, t0, t1, dt

    def _prepare_train_data(self, y0, y, yt, t, t0, t1, dt):
        """
        准备用于训练的数据。

        Parameters:
            y0 (ndarray): 初始状态数据。
            y (ndarray): 目标状态数据。
            yt (ndarray): 目标状态的导数数据。
            t (ndarray): 时间数据。
            t0 (float): 数据起始时间。
            t1 (float): 数据结束时间。
            dt (float): 时间步长。

        Returns:
            tuple: 包含用于训练的 t, y,yt, physics_t 的元组。
        """

        len_t = len(t)
        interval = self.config.interval
        learning_time_len = 1.0
        num_time_block_points = int(learning_time_len / dt)
        num_time_block = int(len_t / num_time_block_points)

        # train_t = t[0 : len_t + 1 : interval].copy()
        train_physics_t = t[0 : len_t // num_time_block + 1 : interval].copy()
        train_y0 = y[0:len_t:num_time_block_points][:-1].copy()

        train_t = []
        train_physics_t = []
        train_y = []
        train_yt = []
        for k in range(num_time_block):
            tmp_t = t[0 : len_t // num_time_block + 1 : interval]
            train_t.append(tmp_t.copy())

            tmp_physics_t = t[0 : len_t // num_time_block + 1 : interval]
            train_physics_t.append(tmp_physics_t.copy())

            tmp_y = y[k * num_time_block_points : (k + 1) * num_time_block_points + 1]
            train_y.append(tmp_y.copy())

            tmp_yt = yt[k * num_time_block_points : (k + 1) * num_time_block_points + 1]
            train_yt.append(tmp_yt.copy())

        train_t = np.concatenate(train_t, axis=0)
        train_physics_t = np.concatenate(train_physics_t, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        train_yt = np.concatenate(train_yt, axis=0)

        return train_y0, train_y, train_yt, train_t, train_physics_t

    def _prepare_val_data(self, y0, y, yt, t, t0, t1, dt):
        val_y = y
        val_yt = yt
        val_t = t
        val_y0 = y0

        len_t = len(t)
        interval = self.config.interval
        learning_time_len = 1.0
        num_time_block_points = int(learning_time_len / dt)
        num_time_block = int(len_t / num_time_block_points)

        val_physics_t = t[0 : len_t // num_time_block + 1 : interval].copy()
        val_y0 = y[0:len_t:num_time_block_points][:-1].copy()

        val_t = []
        val_physics_t = []
        val_y = []
        val_yt = []
        for k in range(num_time_block):
            tmp_t = t[0 : len_t // num_time_block + 1 : interval]
            val_t.append(tmp_t.copy())

            tmp_physics_t = t[0 : len_t // num_time_block + 1 : interval]
            val_physics_t.append(tmp_physics_t.copy())

            tmp_y = y[k * num_time_block_points : (k + 1) * num_time_block_points + 1]
            val_y.append(tmp_y.copy())

            tmp_yt = yt[k * num_time_block_points : (k + 1) * num_time_block_points + 1]
            val_yt.append(tmp_yt.copy())

        val_t = np.concatenate(val_t, axis=0)
        val_physics_t = np.concatenate(val_physics_t, axis=0)
        val_y = np.concatenate(val_y, axis=0)
        val_yt = np.concatenate(val_yt, axis=0)

        return val_y0, val_y, val_yt, val_t, val_physics_t
