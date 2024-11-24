"""
***torchvision库中的数据集的标准化封装*** \n
`tcv` 指的是 `torchvision`
"""


from .dataset import DatasetTemp, _DatasetTemp, uuid_t, DatasetInfo, ModuleInfo, Dataset
from typing import Dict, Any
import torch
import torchvision.datasets as datasets
from hagike.utils.cache import MemCacher


class _TcvDataset(_DatasetTemp):
    """
    二次封装tcv数据集本身，转换独热码并使用内存缓存
    """
    def __init__(self, dataset: Dataset, para: Dict[str, Any]) -> None:
        """
        初始化 \n
        若启用缓存器则需要在 `para` 中定义 `cache_size` 项
        """
        super().__init__(dataset, para)
        if 'cache_size' in self._para:
            self._getter = MemCacher(self._getitem, self._para['cache_size'])
        else:
            self._getter = self._getitem

    def __getitem__(self, index):
        """获取数据"""
        return self._getter(index)

    def _getitem(self, index):
        """数据转换"""
        # torchvision的分类数据集中默认是非独热码，这里需要转换为独热码
        inputs, labels = super().__getitem__(index)
        one_hot = torch.zeros(self._para['num_classes'],
                              device=self._para[ModuleInfo.device], dtype=self._para[ModuleInfo.dtype])
        labels = one_hot.scatter_(0, labels.to(dtype=torch.int64), 1)
        return inputs, labels


class TcvDataset(DatasetTemp):
    """
    torchvision库中的数据集的标准化封装
    """
    def __init__(self, dataset: Any, info: Dict[uuid_t, Any] | None = None) -> None:
        """
        初始化 \n

        :param dataset: tcv数据集，如datasets.CIFAR10 \n
        :param info: `DatasetInfo`，`para` 中无需定义 `train` 参数 \n
        """
        super().__init__(info=info)
        self._container = _TcvDataset
        self._train_dataset = dataset(train=True, **self._info[DatasetInfo.para])
        self._val_dataset = dataset(train=False, **self._info[DatasetInfo.para])

