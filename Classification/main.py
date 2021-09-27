import argparse
import os
from Code.solver import Solver
from Code.data_loader import get_loader
from torch.backends import cudnn
import random
import time
def main(config):
    print("main")

    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net','Swin_Transformer','resnet50','SC_Swin_Transformer']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return
    cur_time = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
    # Create directories if not exist
    config.model_path = os.path.join(config.output_path,config.model_type,cur_time,'model')
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    config.result_path = os.path.join(config.output_path,config.model_type,cur_time,'result')
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    #augmentation_prob= random.random()*0.7
    decay_ratio = 0.95
    decay_epoch = int(config.num_epochs*decay_ratio)

    #config.augmentation_prob = augmentation_prob
    config.num_epochs_decay = decay_epoch

    print(config)

    train_loader = get_loader(
        data_path=config.data_path,
        name_path=config.train_path,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mode='train',
        augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(
        data_path=config.data_path,
        name_path=config.valid_path,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mode='valid',
        augmentation_prob=0.)
    test_loader = get_loader(
        data_path=config.data_path,

        name_path=config.test_path,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mode='test',
        augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=tuple, default=(448,448))
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=6)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--num_epochs_decay', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.5)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)
    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='Swin_Transformer', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/Swin_Transformer/resnet50/SC_Swin_Transformer')
    parser.add_argument('--output_path', type=str, default='/home/ubuntu/liuyiyao/shanghai10yuan_project/Output')
    parser.add_argument('--name_path', type=str, default='/home/ubuntu/liuyiyao/shanghai10yuan_project/Dataset/6_3')#
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/liuyiyao/shanghai10yuan_project/Dataset/shanghai10yuan_valid_slice')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--result_path', type=str, default='')
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--valid_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--cuda_idx', type=int, default=1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = parser.parse_args()
    print("into main")
    config.train_path = config.name_path + "/train/"

    config.valid_path = config.name_path + "/valid/"
    config.test_path = config.name_path + "/test/"


    main(config)

