# -*- coding: utf-8 -*-
# @Author  : LG
# 使用了与torchvision中VGG相同的搭建方式.

from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch

# 参数配置,标准的darknet19参数.
cfg = [32, 'M', 64, 'M', 128, 64, 128, 'M', 256, 128, 256, 'M',
       512, 256, 512, 256, 512, 'M', 1024, 512, 1024, 512, 1024]

cfg_wide = [64, 'M', 128, 'M', 256, 512, 'D', 512, 'M', 512, 256, 512, 'M',
        1024, 512, 1024, 'D', 512, 1024, 'M', 2048, 1024, 1024, 'D', 1024, 1024]

cfg_9 = [32, 'M', 64, 'M', 128, 'M', 256, 'M',
         256, 512, 'M', 512, 512, 1024]

class LabelSmoothing(nn.Module):

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, input, target):
        assert input.size(1) == self.size
        true_dist = input.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(input, Variable(true_dist, requires_grad=False))


class CELoss(nn.Module):

    def __init__(self, class_num=2, label_smooth=None):
        super(CELoss, self).__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        eps = 1e-12
        if self.label_smooth is not None:
            logprobs = F.log_softmax(pred, dim=1)
            target = F.one_hot(target, self.class_num)
            target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
            loss = -1*torch.sum(target*logprobs, 1)
        else:
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))
        return loss.mean()



def make_layers(cfg, in_channels=3, batch_norm=True):
    """
    从配置参数中构建网络
    :param cfg:  参数配置
    :param in_channels: 输入通道数,RGB彩图为3, 灰度图为1
    :param batch_norm:  是否使用批正则化
    :return:
    """
    layers = []
    flag = True             # 用于变换卷积核大小,(True选后面的,False选前面的)
    in_channels= in_channels
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif v == 'D':
            layers.append(nn.Dropout(p=0.3))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=v,
                                    kernel_size=(1, 3)[flag],
                                    stride=1,
                                    padding=(0, 1)[flag],
                                    bias=False))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            in_channels = v

            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        flag = not flag

    return nn.Sequential(*layers)


class Darknet19(nn.Module):
    """
    Darknet19 模型
    """
    def __init__(self, num_classes=2, in_channels=3, batch_norm=True, pretrained=False):
        """
        模型结构初始化
        :param num_classes: 最终分类数       (nums of classification.)
        :param in_channels: 输入数据的通道数  (input pic`s channel.)
        :param batch_norm:  是否使用正则化    (use batch_norm, True or False;True by default.)
        :param pretrained:  是否导入预训练参数 (use the pretrained weight)
        """
        super(Darknet19, self).__init__()
        # 调用nake_layers 方法搭建网络
        # (build the network)
        self.features = make_layers(cfg_9, in_channels=in_channels, batch_norm=batch_norm)
        # 网络最后的分类层,使用 [1x1卷积和全局平均池化] 代替全连接层.
        # (use 1x1 Conv and averagepool replace the full connection layer.)
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1),
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(output_size=(1)),
            # nn.Softmax(dim=0)
        )
        # 导入预训练模型或初始化
        if pretrained:
            self.load_weight()
        else:
            self._initialize_weights()

    def forward(self, x):
        # 前向传播
        x = self.features(x)
        x = self.classifier(x)
        # 为了使输出与使用全连接层格式相同,这里对输出进行了resize
        # resize [B, num_classes, 1, 1] to [B, num_classes]
        x = x.view(x.size(0),-1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weight(self):
        weight_file = 'weights/darknet19-deepBakSu-e1b3ec1e.pth'
        # if not os.path.exists(weight_file):
        #     import wget
        #
        #     url = 'https://s3.ap-northeast-2.amazonaws.com/deepbaksuvision/darknet19-deepBakSu-e1b3ec1e.pth'
        #     print('将下载权重文件,如果太慢,你可以自己下载然后放到weights文件夹下'.format(url))
        #     print('Will download weight file from {}'.format(url))
        #     wget.download(url=url, out='weights/darknet19-deepBakSu-e1b3ec1e.pth')
        # # 转换权重文件中的keys.(change the weights dict `keys)
        # assert len(torch.load(weight_file).keys()) == len(self.state_dict().keys())
        # dic = {}
        # for now_keys, values in zip(self.state_dict().keys(), torch.load(weight_file).values()):
        #     dic[now_keys]=values
        # self.load_state_dict(dic)

if __name__ == '__main__':
    # 原权重文件为1000分类, 在imagenet上进行预训练,
    # Pretrained model train on imagenet dataset. 1000 nums of classifical.
    # top-1 accuracy 76.5% , top-5 accuracy 93.3%.

    net = Darknet19(num_classes=2, pretrained=False)
    print(net)

    x = torch.zeros((2,3,512,512))
    out = net(x)
    print(out.size())
