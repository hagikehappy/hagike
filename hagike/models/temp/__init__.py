"""
模型结构的模板文件 \n
:term \n
    unit - 裸 `nn.Module` 模型 \n
    module_node - 模块节点，最小封装单元，只能包含单一的 `unit`
    module - 模块，仅包含ModuleKey中的固定术语的单一执行流，可包含固定串行化的 `unit` \n
    model_node - 模型节点，由 `module` 或 `module_node` 组成，定义了节点的拓扑结构，作为model的组分不作为单独模型存在 \n
    model - 模型，由若干 `model_node` 组成，各node构成DAG(有向无环图) \n
"""
