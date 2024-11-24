"""
***模型的父亲类模板*** \n
"""


import torch.nn as nn
from typing import Any, Dict, Tuple, List, Mapping, Sequence
from .const import uuid_t
from .error import ModelError, ModelWarning
from .node import ModuleNode
from copy import deepcopy


# 节点标识符，设置为 `str` 是因为 `nn.ModuleDict` 的键值要求为 `str`
node_t = str


class ModelTemp(ModuleNode):
    """模型的通用模板父类"""

    input_node = '_in'
    output_node = '_out'
    _in = 0
    _out = 1
    _blank = 4

    def __init__(self, nodes: Mapping[node_t, ModuleNode],
                 deps: Dict[node_t,  Tuple[List[node_t], List[node_t]]],
                 info: Dict[uuid_t, Any] | None = None) -> None:
        """
        根据输入构建有向无环图方式的模型 \n

        :param nodes: 节点映射表，键值不能与 `input_node` 与 `output_node` 的键值重复 \n
        :param deps: 节点依赖关系表，每个节点都会描述自身的 `(输入节点, 输出节点)`；
                    要求必须含有 `input_node` 与 `output_node` 的键值，
                    前者的 `输入节点` 要求为 `[]`，后者的 `输出节点` 为 `[]` \n
        :param info: 指定初始参数描述，若未输入描述则默认为固定值 \n
        """
        self._deps = deps
        keys, values = nodes.keys(), nodes.values()
        for node in values:
            self.check_node(node)
        if (self.input_node in keys) or (self.output_node in keys):
            raise ModelError(f"{self.input_node} or {self.output_node} should not be used as a key!!!")
        keys = deps.keys()
        if (self.input_node not in keys) or (self.output_node not in keys):
            raise ModelError(f"{self.input_node} or {self.output_node} should be included in deps!!!")
        super().__init__(nn.ModuleDict(nodes), info)
        self._check_model()
        # 临时依赖表，存放依赖关系未满足的节点
        self._dep_dict: Dict[node_t, Tuple[List[node_t] | None, List[node_t] | None]] | None = None
        # 临时等待队列，存放可以进行运算的节点
        self._wait_list: List[node_t] | None = None
        # 数据暂存队列，存储节点的输入数据
        self._store_dict: Dict[node_t, List[Any]] | None = None
        self._store_default = dict()
        for node in self._deps.keys():
            self._store_default[node] = [None for _ in range(len(self._deps[node][self._in]))]

    def _check_model(self) -> None:
        """
        .. todo::
            检查模型合法性，包括其是否成环，以及输入输出数量是否匹配
        """

    def update(self) -> None:
        """
        .. todo::
            更新模型结构
        """

    def _reset_tmp(self) -> None:
        """重置临时结构"""
        self._dep_dict = deepcopy(self._deps)
        self._wait_list = list()
        self._store_dict = deepcopy(self._store_default)

    def _release_tmp(self) -> None:
        """释放临时结构"""
        self._dep_dict = None
        self._wait_list = None
        self._store_dict = None

    def print_model(self, blank: int = 0) -> None:
        """打印模型构成"""
        blank_str = ' ' * blank
        print(f"{blank_str}ModelTemp: {self.__class__.__name__}")
        blank += self._blank
        blank_str += ' ' * self._blank
        for key, node in self._model.items():
            print(f"{blank_str}{key}：")
            node.print_model(self._blank + blank)

    def _step_node(self, this_node: node_t, out_node: List[node_t], results: Sequence[Any]) -> None:
        """处理节点"""
        for i, node in enumerate(out_node):
            # 获取输出位置的输入节点，据此载入至数据暂存区的正确位置
            in_node = self._dep_dict[node][self._in]
            index = in_node.index(this_node)
            in_node.pop(index)
            self._store_dict[node][index] = results[i]
            # 检查该节点是否已经满足依赖条件，若满足则加入等待队列
            if not in_node:
                self._wait_list.append(node)

    def forward(self, *args) -> Tuple | Any:
        """
        :param args: 输入元组，会依次展开给第一级节点
        :return: 输出列表
        """
        # 重置临时结构
        self._reset_tmp()
        outputs = None
        # 解析并遍历入口节点的输出位置
        out_node = self._dep_dict[self.input_node][self._out]
        results = args
        # 遍历输出位置
        self._step_node(self.input_node, out_node, results)
        # 前向传播过程，等待表非空则持续进行下去
        while self._wait_list:
            # 从等待表获取待处理节点，获取后入先出，优先执行最近路径
            this_node = self._wait_list.pop()
            # 检查节点是否为输出节点，如果是则退出循环
            if this_node == self.output_node:
                outputs = self._store_dict[this_node]
                break
            # 处理节点并获取结果
            result = self._model[this_node](*self._store_dict.pop(this_node))
            # 解析当前节点的输出位置
            out_node = self._dep_dict[this_node][self._out]
            results = [result for _ in range(len(out_node))]
            # 遍历输出位置
            self._step_node(this_node, out_node, results)
        # 整理输出结果
        if outputs is None:
            raise ModelError(f"the model output None, something went wrong!!!")
        elif len(outputs) == 1:
            outputs = outputs[0]
        else:
            outputs = tuple(outputs)
        # 释放临时结构
        self._release_tmp()
        return outputs

    def module(self, key: node_t) -> ModuleNode:
        """返回模块"""
        return self._model[key]

    def load_weights(self, weights_src: str | Any, is_path: bool = False, key: node_t = 'all__') -> None:
        """
        根据is_path，选择从路径或从内存中加载指定部分的模块参数 \n
        `key` 指定模块标识符，若为 `all__` 表示整个模型
        """
        if key == 'all__':
            load_weights = super().load_weights
        else:
            load_weights = self._model[key].load_weights
        return load_weights(weights_src, is_path)

    def save_weights(self, path: str | None = None, key: node_t = 'all__') -> Any:
        """
        根据path，选择加载指定部分的模块参数到路径或从内存中 \n
        `key` 指定模块标识符，若为 `all__` 表示整个模型
        """
        if key == 'all__':
            save_weights = super().save_weights
        else:
            save_weights = self._model[key].save_weights
        return save_weights(path)





