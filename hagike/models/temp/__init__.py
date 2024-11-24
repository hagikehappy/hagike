"""
模型结构的模板文件 \n
unit - 裸 `nn.Module` 模型 \n
node - 模块节点，最小封装单元，只能包含单一的 `unit` \n
module - 模块，仅包含ModuleKey中的固定术语的单一执行流，可包含固定串行化的 `unit` \n
model - 模型，由若干 `node` 或 `module` 组成，各node构成DAG(有向无环图) \n
"""


from .model import ModelTemp
from .module import ModuleTemp
from .node import ModuleNode
from .unit import ModuleUnit
from .const import ModuleKey, ModuleInfo, ModuleMode
