"""
***数据集***
"""


import torch
from torch.utils.data import Dataset
from typing import Any, Dict, Tuple
from ..const import uuid_t, DatasetInfo
from ..error import TrainingError
from ...temp.const import ModuleInfo


class _DatasetTemp(Dataset):
    """二次封装Dataset，加入内部配置"""

    def __init__(self, dataset: Dataset, para: Dict[str, Any]) -> None:
        """初始化"""
        self._dataset = dataset
        self._para = para

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取输入并转换设备"""
        inputs, labels = self._dataset[index]
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device=self._para[ModuleInfo.device], dtype=self._para[ModuleInfo.dtype])
        else:
            inputs = torch.tensor(inputs, device=self._para[ModuleInfo.device], dtype=self._para[ModuleInfo.dtype])
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device=self._para[ModuleInfo.device], dtype=self._para[ModuleInfo.dtype])
        else:
            labels = torch.tensor(labels, device=self._para[ModuleInfo.device], dtype=self._para[ModuleInfo.dtype])
        return inputs, labels

    def __len__(self) -> int:
        """获取数据集数量"""
        return len(self._dataset)

    def __getattr__(self, name: str) -> Any:
        """重定向属性或方法至 `self._dataset`"""
        return getattr(self._dataset, name)


class DatasetTemp:
    """
    数据集容器 \n
    由于数据集的处理本身高度多变，因此这里仅给出接口而不包含基本实现
    """

    def __init__(self, train: Dataset | None = None,
                 val: Dataset | None = None,
                 info: Dict[uuid_t, Any] | None = None) -> None:
        """初始化"""
        self._info = DatasetInfo.dict_(info)
        # 基准实现中 train 和 val 直接赋值得到，未用到 self._info[DatasetInfo.para]；但继承类中可能用于创建 train 和 val
        self._train_dataset = train
        self._val_dataset = val
        self._container = _DatasetTemp
        self._train = None
        self._val = None
        self._is_init = False

    def init(self, model_info: Dict[uuid_t, Any]):
        """配置目标设备等"""
        if self._is_init:
            return
        self._is_init = True
        self._info[DatasetInfo.sub_para].update(model_info)
        self._train = self._container(self._train_dataset, self._info[DatasetInfo.sub_para])
        self._val = self._container(self._val_dataset, self._info[DatasetInfo.sub_para])

    @property
    def train(self) -> Dataset:
        if self._is_init:
            return self._train
        else:
            raise TrainingError(f"visiting dataset when DatasetTemp uninitialized!!!")

    @property
    def val(self) -> Dataset:
        if self._is_init:
            return self._val
        else:
            raise TrainingError(f"visiting dataset when DatasetTemp uninitialized!!!")



