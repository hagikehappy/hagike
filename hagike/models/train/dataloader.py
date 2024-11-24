"""
***数据加载器***
"""


from torch.utils.data import DataLoader
from typing import Any, Dict
from .const import uuid_t, DataLoaderInfo
from .dataset import DatasetTemp
from .error import TrainingError


class DataLoaderTemp:
    """数据加载器容器"""

    def __init__(self, dataset: DatasetTemp, info: Dict[uuid_t, Any] | None = None) -> None:
        self._info = DataLoaderInfo.dict_(info)
        self._dataset = dataset
        if self._info[DataLoaderInfo.para]:
            self._info[DataLoaderInfo.train_para] = self._info[DataLoaderInfo.para]
            self._info[DataLoaderInfo.val_para] = self._info[DataLoaderInfo.para]
        self._is_init = False
        self._model_info = None
        self._train = None
        self._val = None

    def init(self, model_info: Dict[uuid_t, Any]):
        """配置目标设备等"""
        if self._is_init:
            return
        self._is_init = True
        self._model_info = model_info
        self._dataset.init(model_info)
        self._train = DataLoader(self._dataset.train, **self._info[DataLoaderInfo.train_para])
        self._val = DataLoader(self._dataset.val, **self._info[DataLoaderInfo.val_para])

    @property
    def train(self) -> DataLoader:
        if self._is_init:
            return self._train
        else:
            raise TrainingError(f"visiting dataloader when DataLoaderTemp uninitialized!!!")

    @property
    def val(self) -> DataLoader:
        if self._is_init:
            return self._val
        else:
            raise TrainingError(f"visiting dataloader when DataLoaderTemp uninitialized!!!")
