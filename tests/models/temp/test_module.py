from hagike.models.temp.model import *
from hagike.models.temp.module import *
from hagike.models.temp.node import *
from torchvision.models import *


def test_models_temp_model():
    """models.temp.model的测试用例"""
    weights = EfficientNet_V2_S_Weights.verify(EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    state_dict = weights.get_state_dict()
    model = ModelTemp(
        {
            'all': ModuleNode(efficientnet_v2_s())
        },
        {
            ModelTemp.input_node: ([], ['all']),
            'all': ([ModelTemp.input_node], [ModelTemp.output_node]),
            ModelTemp.output_node: (['all'], [])
         }
    )
    model.load_weights(state_dict, False, key='all')
    model.to(device='cuda', dtype=torch.float32, mode=False)
    model.print_model()
    model.print_summary()


def test_models_temp_module():
    """models.temp.module的测试用例"""
    # 创建模型
    module_dict = {
        ModuleKey.bone: ModuleNode(efficientnet_v2_l())
    }
    model = ModuleTemp(module_dict)
    # 加载权重
    weights = EfficientNet_V2_L_Weights.verify(EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    state_dict = weights.get_state_dict()
    model.load_weights(state_dict, False, key=ModuleKey.bone)
    model.to(device='cuda', dtype=torch.float32, mode=False)
    # 更新结构
    model.update(ModuleKey.bone, ModuleNode(efficientnet_v2_s()))
    weights = EfficientNet_V2_S_Weights.verify(EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    state_dict = weights.get_state_dict()
    model.load_weights(state_dict, False, key=ModuleKey.bone)
    model.to(device='cuda', dtype=torch.float32, mode=False)
    # 打印信息
    model.print_model()
    model.print_summary()


def test_models_temp_node():
    """models.temp.node的测试用例"""
    weights = EfficientNet_V2_S_Weights.verify(EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    state_dict = weights.get_state_dict()
    unit = efficientnet_v2_s()
    module = ModuleNode(unit)
    module.load_weights(state_dict, False)
    module.to(device='cuda', dtype=torch.float32, mode=False)
    module.print_model()
    module.print_summary()
