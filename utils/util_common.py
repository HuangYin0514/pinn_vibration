import importlib
import random
import time
from functools import wraps

import numpy as np
import torch


def read_config_file(config_file_path):
    spec = importlib.util.spec_from_file_location("config", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def set_random_seed(random_seed=3407):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def tensors_to_numpy(*tensors):
    """
    Convert PyTorch tensors to NumPy arrays.

    Args:
        *tensors: Variable-length input tensors.

    Returns:
        tuple: A tuple containing the NumPy arrays corresponding to the input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].cpu().detach().numpy()
    
    np_arrays = tuple()
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            # 如果是 PyTorch 张量，将其转换为 NumPy 数组
            np_arrays += (tensor.cpu().detach().numpy(),)
        elif isinstance(tensor, np.ndarray):
            # 如果已经是 NumPy 数组，不进行转换
            np_arrays += (tensor,)
        else:
            raise ValueError("Input must be either a PyTorch tensor or a NumPy array.")
    return np_arrays

class lazy_property:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        val = self.func(instance)
        setattr(instance, self.func.__name__, val)
        return val


def save_dict_info(dict_info, file_path):
    with open(file_path, "w") as f:
        for key, value in dict_info.items():
            f.write(f"{key}: {str(value)}\n")


def save_str_info(content, file_path):
    with open(file_path, "w") as f:
        f.write(content)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print(
            "'"
            + func.__name__
            + "'"
            + " took {:.2f} minute ".format((time.time() - t) / 60)
        )
        print("-" * 10)
        return result

    return wrapper


def enable_grad(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.enable_grad():
            result = func(*args, **kwargs)
        return result

    return wrapper


def deprecated(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # print('this function was deprecated!')
        raise Exception(f" {func.__name__} function was deprecated!")

    return wrapper


def find_class_in_module(module_name, class_name):
    """
    在指定模块中查找给定名称的类。

    参数：
        module_name (str): 要查找类的模块的名称。
        class_name (str): 要查找的类的名称。

    返回：
        class_obj (type): 找到的类对象，如果找不到则返回 None。
    """

    try:
        module = importlib.import_module(module_name)
        class_obj = getattr(module, class_name)
        return class_obj
    except (ImportError, AttributeError):
        return None


def initialize_class(module_name,class_name, **kwargs):
    """
    根据给定的类名初始化类的实例对象。

    参数：
        class_name (str): 要初始化的类的名称。
        **kwargs: 要传递给类构造函数的关键字参数。

    返回：
        instance (object): 初始化后的类实例对象。

    异常：
        ValueError: 如果在 'path' 模块中找不到给定的类名，将引发此异常。
    """
    class_obj = find_class_in_module(module_name, class_name)
    if class_obj is not None:
        return class_obj(**kwargs)
    else:
        raise ValueError(f"Class '{class_name}' not found in the 'dynamics' module.")
