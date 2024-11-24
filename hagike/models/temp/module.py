"""
***模块的父类模板*** \n
"""


import torch.nn as nn
from typing import Any, Dict, Tuple, List
from .const import ModuleKey, uuid_t, ModuleMode
from .error import ModelError, ModelWarning
from .node import ModuleNode


class ModuleTemp(ModuleNode):
    """模块的通用模板父类"""

    # 打印时的留空
    _blank = 4

    def __init__(self, nodes: Dict[uuid_t, ModuleNode] | None = None,
                 info: Dict[uuid_t, Any] | None = None) -> None:
        """
        创建结构时的初始化 \n

        :param nodes: 包含所有ModuleNode格式的module字典 \n
        :param info: 信息表 \n
        """
        # _nodes为存储所有key对应部分的容器
        # _model为仅存储所有nn.Module模型的nn容器，每次刷新modules结构都会重建_models
        # _key2index记录了从dict到list的序号映射
        # _mask记录了是否屏蔽其中的某个部分
        # self是可运行的模型
        for node in nodes.values():
            self.check_node(node)
        self._nodes = ModuleKey.dict_(nodes)
        self._mask_dict: Dict[uuid_t, bool] = dict()
        for key in ModuleKey.iter_():
            self._mask_dict[key] = False
        self._mask_list: List[bool] | None = None
        model, self._key2index = self._build_model()
        super().__init__(model, info)

    def _build_model(self) -> Tuple[nn.ModuleList, Dict[uuid_t, int]]:
        """由module构建model和key2index，同时检查类型正确性，刷新掩码表"""
        index = 0
        key2index = dict()
        model = nn.ModuleList()
        self._mask_list = list()
        for key in ModuleKey.iter_():
            node = self._nodes[key]
            if node is not None:
                model.append(node)
                self._mask_list.append(self._mask_dict[key])
                key2index[key] = index
                index += 1
        return model, key2index

    def to_mask(self, key: uuid_t, mask: bool) -> None:
        """改变屏蔽状态，`mask` 为 `True` 时进行屏蔽，否则解屏蔽"""
        self._mask_dict[key] = mask
        self._mask_list[self._key2index[key]] = mask

    def module(self, key: uuid_t) -> ModuleNode:
        """返回模块"""
        return self._nodes[key]

    def print_model(self, blank: int = 0) -> None:
        """打印模型构成"""
        blank_str = ' ' * blank
        print(f"{blank_str}ModuleTemp：{self.__class__.__name__}")
        blank += self._blank
        blank_str += ' ' * self._blank
        for key in ModuleKey.iter_():
            module = self._nodes[key]
            print(f"{blank_str}{ModuleKey.get_name_(key)}：", end='')
            if module is None:
                print("None")
            else:
                print()
                module.print_model(self._blank + blank)

    def forward(self, x):
        """前向传播，若model为空的Sequential则会报错"""
        for i in range(len(self._model)):
            if not self._mask_list[i]:
                x = self._model[i](x)
        return x

    def update(self, key: uuid_t, node: ModuleNode | None):
        """更新模块，key指定模块号"""
        ModuleKey.check_in_(key, all_or_index=False)
        self.check_node(node)
        # 若删除模型结构
        pre_node = self._nodes[key]
        self._nodes[key] = node
        if node is None and pre_node is not None:
            del self._model[self._key2index[key]]
            del self._key2index[key]
        # 若添加模型结构，nn.ModuleList不支持insert操作，因此每次更新时都会重构该这一部分
        elif node is not None and pre_node is not None:
            self._model, self._key2index = self._build_model()
        # 若替换模型结构
        else:
            pass

    def load_weights(self, weights_src: str | Any, is_path: bool = False, key: uuid_t = ModuleKey.all__) -> None:
        """
        根据is_path，选择从路径或从内存中加载指定部分的模块参数 \n
        `key` 指定模块号
        """
        ModuleKey.check_in_(key, all_or_index=True)
        if key == ModuleKey.all__:
            load_weights = super().load_weights
        else:
            load_weights = self._nodes[key].load_weights
        return load_weights(weights_src, is_path)

    def save_weights(self, path: str | None = None, key: uuid_t = ModuleKey.all__) -> Any:
        """
        根据path，选择加载指定部分的模块参数到路径或从内存中 \n
        `key` 指定模块号
        """
        ModuleKey.check_in_(key, all_or_index=True)
        if key == ModuleKey.all__:
            save_weights = super().save_weights
        else:
            save_weights = self._nodes[key].save_weights
        return save_weights(path)
