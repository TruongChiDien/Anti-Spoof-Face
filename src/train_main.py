import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from src.model_lib.MultiFTNet import MultiFTNet
from src.model_lib.MiniFASNet import *

from src.data_io.dataset_loader import get_train_loader

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE,
    'MultiFTNet':MultiFTNet
}

class TrainMain:
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.save_every = conf.save_every
        self.start_epoch = 0
        self.train_loader = get_train_loader(self.conf)


    def train_model(self):
        self._init_model_param()
        self._train_stage()


    def _init_model_param(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self._define_network()
        self.optimizer = optim.SGD(self.model.module.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=5e-4,
                                   momentum=self.conf.momentum)

        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

        print("lr: ", self.conf.lr, '---- epochs: ', self.conf.epochs)


    def _train_stage(self):
        self.model.train()
        self.history = {'loss': [], 'ft_loss': [], 'acc': []}

        for e in range(self.conf.epochs):
            print('Epoch: {} --------- Learning rate: '.format(e, self.schedule_lr.get_lr()[0]))

            r_loss, r_ft_loss, r_acc = 0., 0., 0.
            n_iters = 0

            for sample, ft_sample, target in tqdm(iter(self.train_loader)):
                if self.conf.model_type == 'MultiFTNet':
                    loss, ft_loss, acc = self._train_batch_data(sample, ft_sample, target)
                    r_ft_loss += ft_loss
                else:
                    loss, acc = self._train_batch_data(sample, ft_sample, target)

                r_loss, r_acc += loss, acc
                n_iters += 1
            print(f'\nLoss: {round(r_loss/n_iters, 3)} ---- Acc: {round(r_acc/n_iters, 3)}')

            self.history['loss'].append(r_loss)
            self.history['ft_loss'].append(ft_loss)
            self.history['acc'].append(r_acc)

            self.schedule_lr.step()

        self._save_state()


    def _train_batch_data(self, imgs, ft_imgs, labels):
        self.optimizer.zero_grad()
        labels = labels.to(self.conf.device)
        if self.conf.model_type == 'MultiFTNet':
            embeddings, feature_map = self.model.forward(imgs.to(self.conf.device))
            loss_cls = self.cls_criterion(embeddings, labels)
            loss_ft = self.ft_criterion(feature_map, ft_imgs.to(self.conf.device))
            loss = 0.5*loss_cls + 0.5*loss_ft

        else:
            embeddings = self.model.forward(imgs.to(self.conf.device))
            loss = self.cls_criterion(embeddings, labels)

        acc = self._get_accuracy(embeddings, labels)[0]
        loss.backward()
        self.optimizer.step()

        if self.conf.model_type == 'MultiFTNet':
            return loss_cls.item(), loss_ft.item(), acc.item()
        else:
            return loss.item(), acc.item()


    def _define_network(self):
        param = {
            'num_classes': self.conf.num_classes,
            'img_channel': self.conf.input_channel,
            'embedding_size': self.conf.embedding_size,
            'conv6_kernel': self.conf.kernel_size
        }

        model = MODEL_MAPPING[self.conf.model_type](**param).to(self.conf.device)
        model = torch.nn.DataParallel(model, self.conf.devices)
        model.to(self.conf.device)
        return model


    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret


    def _save_state(self):
        torch.save(self.model.state_dict(), self.conf.save_path)
