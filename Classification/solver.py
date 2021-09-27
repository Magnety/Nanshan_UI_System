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

from .myevaluation import *
from .utils.metrics import DiceLoss
from .arch.network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from .arch.swin_t.build_model import build_swin_t
from .arch.SCTransformer_channel import build_sc_t
import csv
#from .arch.resnet import resnet50
from .arch.vgg import vgg16_bn
from .arch.mv_resnet import resnet50
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.result_path = self.result_path + '/%d-%.6f-%d-%.4f.txt' % (
        self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob)

        with open(Path(self.result_path), "a") as f:
            f.write('%s\n'%str(config))

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
        elif self.model_type == 'SC_Swin_Transformer':
            self.unet = build_sc_t()
        elif self.model_type == 'resnet50':
            self.unet = resnet50()
        elif self.model_type == 'vgg16':
            self.unet = vgg16_bn()

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
        unet_path = os.path.join(self.model_path, '%d-%.6f-%d-%.4f.pkl' % (
        self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.

            for epoch in range(self.num_epochs):

                self.unet.train(True)
                epoch_loss = 0

                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                length = 0

                for i, (images, label, name) in enumerate(self.train_loader):
                    # GT : Ground Truth

                    images = images.to(self.device)
                    # GT = GT.to(self.device)
                    label = label.to(self.device)

                    # SR : Segmentation Result
                    cls = self.unet(images)

                    # print("SR:",SR.shape)
                    # print("label:", label.shape)
                    # print("cls:", cls.shape)
                    #                                   print("GT:",GT.shape)

                    cls_loss = self.cls_criterion(cls, label.squeeze().long())
                    loss = cls_loss
                    epoch_loss += loss.item()

                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()
                    """if not os.path.isdir(os.path.join(self.result_path, str(epoch + 1))):
                        os.makedirs(os.path.join(self.result_path, str(epoch + 1)))

                    for j in range(images.shape[1]):

                        torchvision.utils.save_image(images[:,j:j+1,:,:].data.cpu(),
                                                 os.path.join(self.result_path, str(epoch + 1),
                                                              '%s_train_%d_%d_image.png' % (self.model_type, i,j)))"""

                # Print the log info
                print('Epoch [%d/%d], Loss: %.4f , LR: %.5f' % (
                    epoch + 1, self.num_epochs, \
                    epoch_loss, lr))

                # print()
                # Decay learning rate
                if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (lr / float(self.num_epochs_decay )*5)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                # print ('Decay learning rate to lr: {}.'.format(lr))

                # ===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()
                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                length = 0
                total = 0
                true = 0
                tp = 0
                fp = 0
                tn = 0
                fn = 0
                cls_acc = 0.
                cls_pre = 0.
                cls_rec = 0.
                cls_spe = 0.
                cls_f1 = 0.
                label_list = []
                probility_list = []
                predict_list = []
                name_list = []
                for i, (images, label, name) in enumerate(self.valid_loader):
                    # print(name)
                    epoch_loss=0
                    images = images.to(self.device)
                    # GT = GT.to(self.device)
                    label = label.to(self.device)
                    cls = self.unet(images)

                    """SR2 = torch.cat((SR[0,:,:,:],SR[0,:,:,:],SR[0,:,:,:]),0)
                    SR1 = SR.cpu().detach().numpy()
                    #SR1= np.transpose(SR1,(1,2,0))
                    print(SR1.shape)
                    #scipy.misc.toimage(SR1[0,:,:,:]).save(self.result_path+"/%d_valid.jpeg"%epoch)
                    im = Image.fromarray(np.uint8(SR1[0,0,:,:]*255.0))
                    im.convert('L').save(self.result_path+"/%d_valid.jpeg"%epoch)"""

                    batch_size = label.size(0)

                    cls_cpu = cls.data.cpu().numpy()
                    label_cpu = label.data.cpu().numpy()
                    cls_loss = self.cls_criterion(cls, label.squeeze().long())
                    loss = cls_loss
                    epoch_loss += loss.item()
                    # cls_out = cls_cpu.argmax()
                    for b in range(batch_size):
                        total += 1
                        name_list.append(name[b])
                        probility_list.append(np.exp(cls_cpu[b, 1]))
                        #cls_out = cls_cpu[b].argmax()
                        if np.exp(cls_cpu[b, 1])>0.45:
                            cls_out=1
                        else:
                            cls_out=0
                        label_out = label_cpu[b]
                        label_list.append(label_out)
                        predict_list.append(cls_out)
                        if cls_out == label_out:
                            true += 1
                            if cls_out == 1:
                                tp += 1
                            else:
                                tn += 1
                        else:
                            if cls_out == 1:
                                fp += 1
                            else:
                                fn += 1
                    """if not os.path.isdir(os.path.join(self.result_path, str(epoch + 1))):
                        os.makedirs(os.path.join(self.result_path, str(epoch + 1)))

                    for j in range(images.shape[1]):

                        torchvision.utils.save_image(images[:,j:j+1,:,:].data.cpu(),
                                                 os.path.join(self.result_path, str(epoch + 1),
                                                              '%s_valid_%d_%d_image.png' % (self.model_type, i,j)))"""
                cls_acc = true / total
                cls_pre = tp / (tp + fp + 1e-8)
                cls_rec = tp / (tp + fn + 1e-8)
                cls_spe = tn / (tn + fp + 1e-8)
                cls_f1 = (2 * cls_pre * cls_rec) / (cls_pre + cls_rec + 1e-8)
                unet_score = cls_f1 + cls_acc

                print('[Validation_cls] Loss:%.4f, Acc: %.4f, PRE: %.4f, REC: %.4f, SPE: %.4f, F1: %.4f' % (epoch_loss,
                cls_acc, cls_pre, cls_rec, cls_spe, cls_f1))
                """if not os.path.exists(Path(self.result_path+"/log.txt")):
                    with open(Path(self.result_path+"/log.txt"), "w") as f:
                        print(f)"""
                with open(Path(self.result_path), "a") as f:

                    f.write(('[Validation_cls]Loss:%.4f Acc: %.4f, PRE: %.4f, REC: %.4f, SPE: %.4f, F1: %.4f\n' % (epoch_loss,
                    cls_acc, cls_pre, cls_rec, cls_spe, cls_f1)))
                    f.write(('[Name] %s\n' % str(name_list)))

                    f.write(('[Probility] %s\n' % str(probility_list)))
                    f.write(('[Predict] %s\n' % str(predict_list)))

                    f.write(('[Label] %s\n' % str(label_list)))

                # Save Best U-Net model
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    with open(Path(self.result_path), "a") as f:
                        f.write('Best %s model score : %.4f\n' % (self.model_type, best_unet_score))
                    print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
                    torch.save(best_unet, unet_path)

        # ===================================== Test ====================================#



                self.unet.train(False)
                self.unet.eval()
                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                length = 0
                total = 0
                true = 0
                tp = 0
                fp = 0
                tn = 0
                fn = 0
                cls_acc = 0.
                cls_pre = 0.
                cls_rec = 0.
                cls_spe = 0.
                cls_f1 = 0.
                label_list = []
                probility_list = []
                predict_list = []
                name_list = []
                for i, (images, label, name) in enumerate(self.test_loader):
                    # print(name)
                    images = images.to(self.device)
                    # GT = GT.to(self.device)
                    label = label.to(self.device)
                    cls = self.unet(images)

                    """SR2 = torch.cat((SR[0,:,:,:],SR[0,:,:,:],SR[0,:,:,:]),0)
                    SR1 = SR.cpu().detach().numpy()
                    #SR1= np.transpose(SR1,(1,2,0))
                    print(SR1.shape)
                    #scipy.misc.toimage(SR1[0,:,:,:]).save(self.result_path+"/%d_valid.jpeg"%epoch)
                    im = Image.fromarray(np.uint8(SR1[0,0,:,:]*255.0))
                    im.convert('L').save(self.result_path+"/%d_valid.jpeg"%epoch)"""

                    batch_size = label.size(0)

                    cls_cpu = cls.data.cpu().numpy()
                    label_cpu = label.data.cpu().numpy()

                    # cls_out = cls_cpu.argmax()
                    for b in range(batch_size):
                        total += 1
                        name_list.append(name[b])
                        probility_list.append(np.exp(cls_cpu[b, 1]))
                        # cls_out = cls_cpu[b].argmax()
                        if np.exp(cls_cpu[b, 1]) > 0.5:
                            cls_out = 1
                        else:
                            cls_out = 0
                        label_out = label_cpu[b]
                        label_list.append(label_out)
                        predict_list.append(cls_out)
                        if cls_out == label_out:
                            true += 1
                            if cls_out == 1:
                                tp += 1
                            else:
                                tn += 1
                        else:
                            if cls_out == 1:
                                fp += 1
                            else:
                                fn += 1
                    """if not os.path.isdir(os.path.join(self.result_path, str(epoch + 1))):
                        os.makedirs(os.path.join(self.result_path, str(epoch + 1)))

                    for j in range(images.shape[1]):

                        torchvision.utils.save_image(images[:,j:j+1,:,:].data.cpu(),
                                                 os.path.join(self.result_path, str(epoch + 1),
                                                              '%s_valid_%d_%d_image.png' % (self.model_type, i,j)))"""
                cls_acc = true / total
                cls_pre = tp / (tp + fp + 1e-8)
                cls_rec = tp / (tp + fn + 1e-8)
                cls_spe = tn / (tn + fp + 1e-8)
                cls_f1 = (2 * cls_pre * cls_rec) / (cls_pre + cls_rec + 1e-8)
                unet_score = cls_f1 + cls_acc

                print('[      Test_cls] Acc: %.4f, PRE: %.4f, REC: %.4f, SPE: %.4f, F1: %.4f' % (
                    cls_acc, cls_pre, cls_rec, cls_spe, cls_f1))
                """if not os.path.exists(Path(self.result_path+"/log.txt")):
                    with open(Path(self.result_path+"/log.txt"), "w") as f:
                        print(f)"""
                with open(Path(self.result_path), "a") as f:
                    f.write(('[Test_cls] Acc: %.4f, PRE: %.4f, REC: %.4f, SPE: %.4f, F1: %.4f\n' % (
                        cls_acc, cls_pre, cls_rec, cls_spe, cls_f1)))
                    f.write(('[Name] %s\n' % str(name_list)))
                    f.write(('[Probility] %s\n' % str(probility_list)))
                    f.write(('[Predict] %s\n' % str(predict_list)))
                    f.write(('[Label  ] %s\n' % str(label_list)))
                    f.write(('\n' ))

