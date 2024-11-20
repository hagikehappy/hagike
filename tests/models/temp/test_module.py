import torch

from hagike.models.temp.module import *
from hagike.models.temp.module_node import *
from torchvision.models import *


def test_models_temp_module():
    """models.temp.module的测试用例"""
    module_dict = {
        ModuleKey.all__: efficientnet_v2_s()
    }
    weights = EfficientNet_V2_S_Weights.verify(EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    state_dict = weights.get_state_dict()
    module = ModuleTemp(module_dict)
    module.load_weights(ModuleKey.all__, state_dict, False)
    module.trans_para(device='cuda')
    module.print_summary()


def test_models_temp_module_node():
    """mode;s.temp.module_node的测试用例"""
    weights = EfficientNet_V2_S_Weights.verify(EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    state_dict = weights.get_state_dict()
    unit = efficientnet_v2_s()
    module = ModuleNode(unit)
    module.load_weights(state_dict, False)
    module.to(device='cuda', dtype=torch.float32, is_train=False)
    module.print_summary()
