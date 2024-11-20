import os.path
import subprocess
import shutil

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def display_graph(graph_t, channel, save_path=None):
    """显示4D输入图"""
    if save_path is not None:
        plt.imsave(save_path, graph_t[0, channel, :, :], cmap='gray')
    plt.imshow(graph_t[0, channel, :, :], cmap='gray')
    plt.show()


class Train_Rfseq:
    """UNet的训练器封装类"""

    def __init__(self, start_detect=False, start_monitor=True, rf_or_label=True, backbone=None,
                 train_from_exist=None, xy_or_heatmap=True):
        """训练器初始化"""
        torch.autograd.set_detect_anomaly(start_detect)
        torch.manual_seed(int(time.time()))

        self.device = settings[Config_Item.device]
        self.rf_or_label = rf_or_label
        self.xy_or_heatmap = xy_or_heatmap

        if not rf_or_label:
            if xy_or_heatmap:
                if settings[Config_Item.label_loss_type] == Loss_Type.L1_Loss.name:
                    self.label_criterion = nn.L1Loss()
                elif settings[Config_Item.label_loss_type] == Loss_Type.L2_Loss.name:
                    self.label_criterion = nn.MSELoss()
                else:
                    raise abort.TrainAbort("Wrong Loss Type!!!")
            else:
                if settings[Config_Item.label_loss_type] == Loss_Type.JointsMSELoss.name:
                    self.label_criterion = JointsMSELoss(target_weight=settings[Config_Item.loss_para]['p_weight'])
                else:
                    raise abort.TrainAbort("Wrong Loss Type!!!")

        if xy_or_heatmap:
            if settings[Config_Item.loss_type] == Loss_Type.Rfseq_Static_Loss.name:
                self.rf_criterion = Rfseq_Static_Loss(
                    hyperpara=settings[Config_Item.loss_para]['hyperpara'],
                    point_num=settings[Config_Item.seq_len],
                    in_size=settings[Config_Item.data_crop_size],
                    p_weight=settings[Config_Item.loss_para]['p_weight']
                )
            else:
                raise abort.TrainAbort("Wrong Loss Type!!!")
        else:
            # 待开发
            self.rf_criterion = self.label_criterion

        if xy_or_heatmap:
            self.model = Rfseq_Static(settings[Config_Item.model_width], settings[Config_Item.model_depth],
                                      in_channels=1, out_channels=2,
                                      in_size=settings[Config_Item.data_crop_size], out_len=settings[Config_Item.seq_len],
                                      backbone=backbone)
        else:
            hr_cfg = load_config(settings[Config_Item.model_heatmap_cfg])
            self.model = PoseHighResolutionNet(cfg=hr_cfg)
        if train_from_exist is not None:
            self.model.load_state_dict(torch.load(train_from_exist, map_location='cpu'))
        self.model = self.model.to(self.device)

        if settings[Config_Item.train_from_exist] is True:
            saved_state_dict = torch.load(settings[Config_Item.train_load_path], map_location=self.device)
            self.model.load_state_dict(saved_state_dict)

        if settings[Config_Item.optimizer_type] == Optimizer_Type.SGD.name:
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=settings[Config_Item.learning_rate],
                                       momentum=settings[Config_Item.optimizer_para]['momentum'],
                                       dampening=settings[Config_Item.optimizer_para]['dampening'],
                                       weight_decay=settings[Config_Item.optimizer_para]['weight_decay'])
        elif settings[Config_Item.optimizer_type] == Optimizer_Type.Adam.name:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=settings[Config_Item.learning_rate],
                                        betas=settings[Config_Item.optimizer_para]['betas'])
        else:
            raise abort.TrainAbort("Wrong Optimizer!!!")

        self.total_epochs = settings[Config_Item.total_epochs]

        if rf_or_label:
            self.train_dataset = Rfseq_Static_Dataset(mode=True,
                                                      root_dir=settings[Config_Item.dataset_final_cache_path],
                                                      rate_scale=(0.0, 0.1), rf_or_label=True,
                                                      xy_or_heatmap=xy_or_heatmap)
            self.train_dataloader = DataLoader(self.train_dataset,
                                               batch_size=settings[Config_Item.train_batch_size], shuffle=True)
            self.eval_dataset = Rfseq_Static_Dataset(mode=True, root_dir=settings[Config_Item.dataset_final_cache_path],
                                                     rate_scale=(0.1, 0.12), xy_or_heatmap=xy_or_heatmap)
            self.eval_dataloader = DataLoader(self.eval_dataset,
                                              batch_size=settings[Config_Item.train_batch_size], shuffle=True)
        else:
            self.train_dataset = Rfseq_Static_Dataset(mode=True,
                                                      root_dir=settings[Config_Item.dataset_final_cache_path],
                                                      rate_scale=(0, 0.2), rf_or_label=False,
                                                      label_path=settings[Config_Item.dataset_label_cache_path],
                                                      xy_or_heatmap=xy_or_heatmap,
                                                      output_size=settings[Config_Item.data_crop_size] // 4,
                                                      input_size=settings[Config_Item.data_crop_size])
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False)
            self.eval_dataset = copy.deepcopy(self.train_dataset)
            self.eval_dataloader = copy.deepcopy(self.train_dataloader)

        now_date = datetime.datetime.now()
        formatted_date = now_date.strftime("%Y-%m-%d-%H-%M-%S")
        self.model_save_path = os.path.join(settings[Config_Item.model_save_path], formatted_date)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        else:
            raise abort.TrainAbort("Model Save Dir EXIST!!!")

        try:
            shutil.copy2(settings[Config_Item.config_json_path], self.model_save_path)
        except FileNotFoundError:
            raise abort.TrainAbort("The config.json doesn't EXIST!!!")

        self.best_eval_score = None
        self.best_model_path = None

        ## 配置tensorboard监控
        self.monitor_path = os.path.join(self.model_save_path, settings[Config_Item.monitor_dir_name])
        os.makedirs(self.monitor_path)
        self.monitor = SummaryWriter(log_dir=self.monitor_path)
        ## 可视化结构模型
        data = torch.load('cache/final_crop/CPU/0000/unet_crop/0001/000000').to(self.device)
        self.monitor.add_graph(self.model, data.unsqueeze(0).unsqueeze(0))
        self.monitor.add_custom_scalars_multilinechart(['Train Loss', 'Eval  Loss'], title='Losses')
        ## 用于在终端启动独立的tensorboard进程
        if start_monitor:
            self.monitor_process = subprocess.Popen(
                [f'konsole --hold -e tensorboard --logdir={self.monitor_path}'], shell=True, close_fds=True)

    def train_all_epochs(self):
        """全序数训练器"""
        counter_t = DynamicCounter(self.total_epochs, "Training Progress: ", 1)
        for epoch in range(1, 1 + self.total_epochs):
            counter_t.increment()
            self.train_one_epoch(epoch)
        self.monitor.close()

    def train_one_epoch(self, epoch):
        """一个训练循环"""

        ## 训练模式
        total_train_loss = 0
        self.model.train()
        train_times = 1
        for inputs, indexes, labels in tqdm(self.train_dataloader):
            ## 前向传播
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            if self.rf_or_label:
                loss = self.rf_criterion(outputs, inputs.squeeze(1))  # 去除通道维度变为掩码
            else:
                labels = labels.to(self.device)
                loss = self.label_criterion(outputs, labels)
            try:
                if torch.isnan(loss).any():
                    raise ValueError
            except ValueError:
                print()
                print(self.train_dataset.file_list[indexes[0]])
                print()
                self.optimizer.zero_grad()
                continue
            train_times += 1
            total_train_loss += loss
            ## 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        total_train_loss /= train_times

        ## 评估模式
        total_eval_loss = 0
        # self.model.eval()
        eval_times = 1
        with torch.no_grad():
            for inputs, indexes, labels in tqdm(self.eval_dataloader):
                ## 前向传播
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                # display_graph(inputs.to('cpu'), 0)
                # for i in range(3):
                #     display_graph(labels.to('cpu'), i)
                # self._transfer_result_graph(inputs, labels, display=True)
                # for i in range(3):
                #     display_graph(outputs.to('cpu'), i)
                # self._transfer_result_graph(inputs, outputs, display=True)
                if self.rf_or_label:
                    loss = self.rf_criterion(outputs, inputs.squeeze(1))  # 去除通道维度变为掩码
                else:
                    labels = labels.to(self.device)
                    loss = self.label_criterion(outputs, labels)
                try:
                    if torch.isnan(loss).any():
                        raise ValueError
                except ValueError:
                    print()
                    print(self.eval_dataset.file_list[indexes[0]])
                    print()
                    continue
                eval_times += 1
                total_eval_loss += loss
        total_eval_loss /= eval_times
        ## 打印训练信息
        print("       Epoch [{}/{}], Train Loss: {:.4f}, Eval Loss: {:.4f}, "
              .format(epoch, self.total_epochs, total_train_loss, total_eval_loss))

        ## 保存模型参数
        if self.best_eval_score is None or self.best_eval_score > total_eval_loss:
            save_name = ("Rfseq_Static_epoch_{}_train_loss_{:.4f}_eval_loss_{:.4f}.pth"
                         .format(epoch + 1, total_train_loss, total_eval_loss))
            save_path = os.path.join(self.model_save_path, save_name)
            torch.save(self.model.state_dict(), save_path)
            self.best_eval_score = total_eval_loss
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            self.best_model_path = save_path

        ## 添加数值监视
        self.monitor.add_scalar('Train Loss', total_train_loss, global_step=epoch)
        self.monitor.add_scalar('Eval  Loss', total_eval_loss, global_step=epoch)
        ## 添加图像效果监视
        self._monitor_add_graph(epoch, True)
        self._monitor_add_graph(epoch, False)

    def _transfer_result_graph(self, input_t, output_t, display=False):
        """显示结果图"""
        if self.xy_or_heatmap:
            x_list, y_list = self.rf_criterion.output2xylist(output_t)
        else:
            x_list, y_list = heatmap2xy(output_t)
            x_list, y_list = (x_list * 4).to(torch.int32), (y_list * 4).to(torch.int32)
        x_list, y_list = x_list[0], y_list[0]
        dot_round = settings[Config_Item.monitor_graph_dot_round]
        length = len(x_list)
        size = int(input_t.shape[-1])
        for i in range(length):
            x_min = x_list[i] - dot_round
            x_min = 0 if x_min < 0 else x_min
            x_max = x_list[i] + dot_round
            x_max = size if x_max >= size else x_max
            y_min = y_list[i] - dot_round
            y_min = 0 if y_min < 0 else y_min
            y_max = y_list[i] + dot_round
            y_max = size if y_max >= size else y_max
            input_t.squeeze()[x_min: x_max, y_min: y_max] = \
                (length - 1 - i) / (length - 1) * 0.6 + 0.2
        if display:
            display_graph(input_t.to('cpu'), 0)
        return input_t

    def _monitor_add_graph(self, epoch, mode):
        """添加测试图"""
        if mode:
            input_t = self.train_dataset[0][0].unsqueeze(0).to(self.device)
            description = 'Train Graph'
        else:
            input_t = self.eval_dataset[0][0].unsqueeze(0).to(self.device)
            description = 'Eval  Graph'

        self.model.train()
        output_t = self.model(input_t)
        result_t = self._transfer_result_graph(input_t, output_t)

        result_t = F.interpolate(result_t, size=settings[Config_Item.monitor_graph_display_size],
                                mode='bilinear', align_corners=False)
        self.monitor.add_image(description, result_t.squeeze(0),
                               global_step=epoch, dataformats="CHW")
