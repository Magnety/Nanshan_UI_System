import os
import numpy as np
import time
import datetime
import torchvision
from pathlib import Path
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.backends import cudnn
import cv2
import matplotlib.pyplot as plt

from myevaluation import *
from utils.metrics import DiceLoss
from arch.network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from swin_t.build_model import build_swin_t
from multi_view.mv_swin_t.build_model import build_mv_swin_t
import csv
# from arch.resnet import resnet50
from arch.mv_resnet import resnet50
from PIL import Image
import scipy.misc

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
        self.dc_criterion = DiceLoss()
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.curtime1 = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.result_path = self.result_path + '/%s-%d-%.4f-%d-%.4f-%s' % (
        self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob,self.curtime1)


        self.t = config.t
        self.build_model()

    def build_model(self):
        torch.cuda.empty_cache()
        np.random.seed(12345)
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=9, output_ch=1)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=9, output_ch=1, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=9, output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=9, output_ch=1, t=self.t)
        elif self.model_type == 'Swin_Transformer':
            self.unet = build_swin_t()
        elif self.model_type == 'mv_Swin_Transformer':
            self.unet = build_mv_swin_t()
        elif self.model_type == 'resnet50':
            self.unet = resnet50()

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)

    # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)
        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#

        unet_path = "/home/ubuntu/liuyiyao/Multi_modal_Image/multi_view/new_models/bmv_1_aug_train/full/Breast_multiview_batch_3/Swin_Transformer/Swin_Transformer-3000-0.0000-2700-0.5000-20210723-231511.pkl"
        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
            del self.unet
            self.build_model()
            self.unet.load_state_dict(torch.load(unet_path))
            self.unet.train(False)
            self.unet.eval()
            for i, (images, label, name) in enumerate(self.valid_loader):

                print(i)
                # print(name)

                images = images.to(self.device)
                # GT = GT.to(self.device)
                label = label.to(self.device)
                target_index = None
                print(images.shape)
                #mask = grad_cam(images, target_index)
                x = self.unet.patch_embed(images)
                #x = x + self.unet.absolute_pos_embed
                for layer in self.unet.layers:
                    x = layer(x)
                    features = x

                x = self.unet.norm(x)  # B L C
                x = self.unet.avgpool(x.transpose(1, 2))  # B C 1

                x = torch.flatten(x, 1)
                #model.eval()

                #features =self.unet.layers(x)
                output = self.unet.head(x)

                # 为了能读取到中间梯度定义的辅助函数
                def extract(g):
                    global features_grad
                    features_grad = g

                # 预测得分最高的那一类对应的输出score
                pred = torch.argmax(output).item()
                pred_class = output[:, pred]
                features.register_hook(extract)
                pred_class.backward()  # 计算梯度
                grads = features_grad  # 获取梯度
                features = features.transpose(1, 2)
                features = features.view(1,768,4,4)
                grads =  grads.transpose(1, 2)
                grads= grads.view(1,768,4,4)
                print("grades.shape:",grads.shape)
                #grads.

                pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1,1))
                #print(" pooled_grads.shape:",  pooled_grads.shape)
                # 此处batch size默认为1，所以去掉了第0维（batch size维）
                pooled_grads = pooled_grads[0]
                features = features[0]
                # 512是最后一层feature的通道数
                print("features.shape:",features.shape)
                for i in range(768):
                    features[i,...] *= pooled_grads[i,...]

                # 以下部分同Keras版实现
                heatmap = features.data.cpu().detach().numpy()
                heatmap = np.mean(heatmap, axis=0)

                heatmap = np.maximum(heatmap, 0)
                heatmap /= np.max(heatmap)

                # 可视化原始热力图

                plt.matshow(heatmap)
                plt.show()
                img_path = '../dataset/get_pseudo_3d_view_slices/' + name[0]+'/img/1.bmp'
                save_path = '/home/ubuntu/liuyiyao/Multi_modal_Image/grad_cam/'+name[0]+'.bmp'
                img = cv2.imread(img_path)  # 用cv2加载原始图像
                heatmap = cv2.resize(heatmap, (128, 128))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
                superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
                cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘

                #cls = self.unet(images)






                """SR2 = torch.cat((SR[0,:,:,:],SR[0,:,:,:],SR[0,:,:,:]),0)
                SR1 = SR.cpu().detach().numpy()
                #SR1= np.transpose(SR1,(1,2,0))
                print(SR1.shape)
                #scipy.misc.toimage(SR1[0,:,:,:]).save(self.result_path+"/%d_valid.jpeg"%epoch)
                im = Image.fromarray(np.uint8(SR1[0,0,:,:]*255.0))
                im.convert('L').save(self.result_path+"/%d_valid.jpeg"%epoch)"""




        # ===================================== Test ====================================#



