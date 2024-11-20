"""
***数据包封装*** \n
.. todo::
    完善压缩类型 \n
"""

from .const import *
from .error import *
from typing import Mapping, Any, Sequence
from hagike.utils.file import *
from hagike.utils.cache import *


class DataPackTemp:
    """数据包模板"""

    def __init__(self, data: Any, src: uuid_t = DataSrc.mem, form: uuid_t = DataForm.direct,
                 info: Dict[uuid_t, Any] = None) -> None:
        """
        创建数据包

        :param data 数据描述符，根据 `src` 进行解析 \n
        :param src 数据源类型
        :param form 数据格式
        :param info 数据额外信息
        """
        DataSrc.check_in_(src), DataForm.check_in_(form)
        info = DataInfo.dict_(info)
        self._data, self._src, self._form, self._info = data, src, form, info

    @property
    def data(self) -> Any:
        self.use()
        return self._data

    @data.setter
    def data(self, data: Any) -> None: self._data = data
    @property
    def is_mem(self) -> bool: return self._src == DataSrc.mem
    @property
    def is_direct(self) -> bool: return self._form == DataForm.direct
    @property
    def is_usable(self) -> bool: return self.is_direct and self.is_mem

    def check(self) -> None:
        """确认数据可达性，若不可达则报错，如果 `_src` 类型为 `else__`，则要求在 `_info` 中自定义检查函数"""
        if self._src == DataSrc.mem:
            return
        elif self._src == DataSrc.file:
            check_path_readable(self._data)
        elif self._src == DataSrc.else__:
            func = self._info[DataInfo.load_check]
            if func is not None:
                func(self._data)
            else:
                raise DataPackError(f"Check func for Data src type 'else__' is not included in info")
        else:
            raise DataPackError(f"Check func for src type {DataSrc.get_name_(self._src)} is not implemented")

    def use(self) -> None:
        """使数据直接可用"""
        self.load()
        self.transfer()

    def load(self) -> None:
        """加载数据包至内存"""
        self.check()
        if self._src == DataSrc.mem:
            return
        elif self._src == DataSrc.file:
            self._data = load_data_from_pkl(self._data)
        elif self._src == DataSrc.else__:
            func = self._info[DataInfo.load_func]
            if func is not None:
                self._data = func(self._data)
            else:
                raise DataPackError(f"Load func for Data src type 'else__' is not included in info")
        else:
            # todo: 压缩类型的加载
            raise DataPackError(f"Load func for src type {DataSrc.get_name_(self._src)} is not implemented")
        self._src = DataSrc.mem

    def transfer(self) -> None:
        """转换至直接可用格式"""
        if self._form == DataForm.direct:
            return
        elif self._form == DataForm.mem_cached:
            func = self._info[DataInfo.mem_cacher]
            if func is not None:
                self._data = func(self._data)
            else:
                raise DataPackError(f"Transfer func for Data form 'mem_cached' is not included in info")
        elif self._form == DataForm.else__:
            func = self._info[DataInfo.form_func]
            if func is not None:
                self._data = func(self._data)
            else:
                raise DataPackError(f"Transfer func for Data form 'else__' is not included in info")
        else:
            raise DataPackError(f"Check func for form {DataSrc.get_name_(self._src)} is not implemented")
        self._form = DataForm.direct

    def to(self) -> None:
        """
        .. todo::
            转换数据源或格式
        """


