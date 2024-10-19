"""
数据缓存类
"""


import os
import pickle
import torch
from PIL import Image
from torchvision import transforms
from typing import Any, Callable
from functools import lru_cache


def image_to_tensor(image_path: str) -> torch.Tensor:
    """图片转换为张量"""
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')
    return transform(image)


def save_data_to_pkl(data: Any, file_path: str) -> None:
    """将数据缓存为pkl格式"""
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_data_from_pkl(file_path: str) -> Any:
    """将数据从pkl加载"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


class Mem_Cacher:
    """内存缓存器"""
    def __init__(self, func: Callable, max_size: int, typed: bool):
        self.func = func
        self.cached_func = lru_cache(maxsize=max_size, typed=typed)(self.func)

    def __call__(self, *args, **kwargs):
        return self.cached_func(*args, **kwargs)


