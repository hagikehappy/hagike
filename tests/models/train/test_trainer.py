from hagike.models.train import *


def test_models_train_trainer():
    """测试models.train.trainer"""

    import torchvision.models as models
    import torchvision.transforms as transforms
    from hagike.models.train.dataset.tcv import TcvDataset, datasets
    from torch.optim import Adam
    from hagike.models.temp import ModuleTemp, ModuleKey
    from hagike.models.temp.common import ModuleTemp_MaskHead
    import torch.nn as nn
    import torch

    # 定义数据
    transform = transforms.Compose([transforms.ToTensor()])
    tcv_dataset = TcvDataset(
        dataset=datasets.CIFAR10,
        info={DatasetInfo.para: {'root': 'data/torchvision', 'download': True, 'transform': transform},
              DatasetInfo.sub_para: {'num_classes': 10, 'cache_size': 60000}}
    )
    loader = DataLoaderTemp(tcv_dataset, info={
        DataLoaderInfo.para: {'batch_size': 32, 'shuffle': True}
    })

    # 定义模型、优化器
    model = ModuleTemp_MaskHead(nodes={
        ModuleKey.bone: ModuleNode(models.resnet18(num_classes=10)),
        ModuleKey.head: ModuleNode(nn.Softmax(dim=-1))
    })

    # 定义其它
    optim = OptimTemp(info={OptimInfo.op_type: Adam, OptimInfo.para: {}, OptimInfo.lr: 0.001})
    criterion = CriterionTemp(info={CriterionInfo.crt_type: torch.nn.CrossEntropyLoss, CriterionInfo.para: {}})
    monitor = TrainerMonitor(info={MonitorInfo.logdir: 'tmp/logs', MonitorInfo.autostart: True})

    # 定义评估方式
    evaluator = EvaluatorTemp(info={EvaluatorInfo.effector_type: EffectorTemp, EvaluatorInfo.para: {}})

    # 定义训练器
    trainer = TrainerTemp(
        model, optim, criterion, loader, monitor, evaluator,
        info={TrainerInfo.device: 'cuda', TrainerInfo.max_epochs: 30}
    )
    trainer.train()
    trainer.end()

