"""
***数据缓存类*** \n
"""


import os
import pickle
from typing import Any, Callable
from functools import lru_cache
from .file import *


def save_data_to_pkl(data: Any, path: str) -> None:
    """将数据缓存为pkl格式"""
    ensure_path_writable(path)
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_data_from_pkl(path: str) -> Any:
    """将数据从pkl加载"""
    check_path_readable(path)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


class Mem_Cacher:
    """内存缓存器"""
    def __init__(self, func: Callable, max_size: int, typed: bool):
        self.func = func
        self.cached_func = lru_cache(maxsize=max_size, typed=typed)(self.func)

    def __call__(self, *args, **kwargs):
        return self.cached_func(*args, **kwargs)


