"""
训练器
"""


from hagike.utils.enum import *
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import ultralytics.models


class TrainerTemp:
    """基本训练器"""

    def __init__(self, model: nn.Module):
        """初始化，输入模型、数据加载器、训练参数"""


