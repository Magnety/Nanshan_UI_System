import os
import argparse
import shutil
from shutil import copyfile
from .misc import printProgressBar
import random as rd

def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        #shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

def main(config):
    _test_idx = [1, 5, 6, 11, 12, 18, 22, 27, 28, 29, 30, 31,
                32, 34, 40, 44, 47, 48, 51, 52, 53, 56, 57,
                58, 59, 63, 68, 70, 74, 81, 84, 85, 90, 96,
                97, 99, 102, 104, 109, 114, 121, 122, 125, 128, 131,
                134, 137, 138, 139, 140, 141, 142, 143, 147, 149, 152]
    rd.shuffle(_test_idx)

    test_idx = []
    for i in range(5):
        test_idx1 = []
        if i==0:
            for j in range(12):
                idx = _test_idx.pop()
                test_idx1.append(idx)
        else:
            for j in range(11):
                idx = _test_idx.pop()
                test_idx1.append(idx)
        test_idx.append(test_idx1)


    import random
    print(len(test_idx))
    for i in range(len(test_idx)):
        train = []
        test = []
        test_tmp = []
        for l in range(len(test_idx)):
            if l != i:
                test_tmp.extend(test_idx[l])
        # print(test_tmp)
        for j in range(153):
            if j not in test_idx[i]:
                train.append(j)
        for k in train:
            if k not in test_tmp:
                test.append(k)
        if i < 1:
            test1 = random.sample(test, 19)
            # print(test1)
            test_idx[i].extend(test1)
        elif i < 3:
            test1 = random.sample(test, 20)
            # print(test1)
            test_idx[i].extend(test1)
        else:
            test1 = random.sample(test, 19)
            # print(test1)
            test_idx[i].extend(test1)
        test_idx[i].sort()
    train_idx = []

    for i in range(len(test_idx)):
        train = []
        # print(test_idx[i])
        # print()
        for j in range(153):
            if j not in test_idx[i]:
                # print(j)
                train.append(j)
        # print(train)
        train_idx.append(train)
    print(train_idx[0])
    print(test_idx)
    filenames = os.listdir(config.origin_data_path)
    data_list = []
    #GT_list = []
    for filename in filenames:
        data_list.append(filename)
    print(data_list)
    for fold in range(5):
        rm_mkdir(config.path+'/Breast_multiview_batch_%d/train'%fold)
        #rm_mkdir(config.path+'/Breast_multiview_batch_%d/train_GT'%fold)
        rm_mkdir(config.path+'/Breast_multiview_batch_%d/valid'%fold)
        #rm_mkdir(config.path+'/Breast_multiview_batch_%d/valid_GT'%fold)
        #rm_mkdir(config.path+'/Breast_multiview_batch_%d/test'%fold)
        #rm_mkdir(config.path+'/Breast_multiview_batch_%d/test_GT'%fold)
        train_path = config.path+'/Breast_multiview_batch_%d/train'%fold
        valid_path = config.path+'/Breast_multiview_batch_%d/valid'%fold
        print('\nNum of train set : ',len(train_idx[fold]))
        print('\nNum of valid set : ',len(test_idx[fold]))
        #print('\nNum of test set : ',num_test)



        for idx in train_idx[fold]:
            src = config.origin_data_path+'/'+ data_list[idx]+'/img'
            rm_mkdir(train_path + '/' + data_list[idx] + '/img')
            rm_mkdir(train_path + '/' + data_list[idx] + '/mask')
            jpg_names = os.listdir(src)
            for jpg_name in jpg_names:
                img_src = config.origin_data_path+'/'+ data_list[idx]+'/img'+'/'+jpg_name
                img_dst = train_path+'/'+data_list[idx]+'/img/'+jpg_name
                copyfile(img_src, img_dst)
                mask_src = config.origin_data_path+'/'+ data_list[idx]+'/mask'+'/'+jpg_name
                mask_dst = train_path+'/'+data_list[idx]+'/mask/'+jpg_name
                copyfile(mask_src, mask_dst)
                label_src = config.origin_data_path+'/'+ data_list[idx]+'/label.txt'
                label_dst = train_path+'/'+data_list[idx]+'/label.txt'
                copyfile(label_src, label_dst)

            printProgressBar(i + 1, len(train_idx[fold]), prefix = 'Producing train set:', suffix = 'Complete', length = 50)


        for idx in test_idx[fold]:
            src = config.origin_data_path + '/' + data_list[idx] + '/img'
            jpg_names = os.listdir(src)
            rm_mkdir(valid_path + '/' + data_list[idx] + '/img')
            rm_mkdir(valid_path + '/' + data_list[idx] + '/mask')
            for jpg_name in jpg_names:

                img_src = config.origin_data_path + '/' + data_list[idx] + '/img' + '/' + jpg_name
                img_dst = valid_path + '/' + data_list[idx] + '/img/' + jpg_name
                copyfile(img_src, img_dst)
                mask_src = config.origin_data_path + '/' + data_list[idx] + '/mask' + '/' + jpg_name
                mask_dst = valid_path + '/' + data_list[idx] + '/mask/' + jpg_name
                copyfile(mask_src, mask_dst)
                label_src = config.origin_data_path + '/' + data_list[idx] + '/label.txt'
                label_dst = valid_path + '/' + data_list[idx] + '/label.txt'
                copyfile(label_src, label_dst)

            printProgressBar(i + 1, len(test_idx[fold]), prefix = 'Producing valid set:', suffix = 'Complete', length = 50)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    # data path
    parser.add_argument('--origin_data_path', type=str, default=r'/home/ubuntu/liuyiyao/Multi_modal_Image/dataset/get_pseudo_3d_view_resample')
    parser.add_argument('--origin_GT_path', type=str, default='../ISIC/dataset/ISIC2018_Task1_Training_GroundTruth')
    parser.add_argument('--path', type=str, default='../dataset/bmv_5')
    """parser.add_argument('--train_path', type=str, default='../dataset/Breast_multiview_b1/train/')
    parser.add_argument('--train_GT_path', type=str, default='../dataset/Breast_multiview_b1/train_GT/')
    parser.add_argument('--valid_path', type=str, default='../dataset/Breast_multiview_b1/valid/')
    parser.add_argument('--valid_GT_path', type=str, default='../dataset/Breast_multiview_b1/valid_GT/')
    parser.add_argument('--test_path', type=str, default='../dataset/Breast_multiview_b1/test/')
    parser.add_argument('--test_GT_path', type=str, default='../dataset/Breast_multiview_b1/test_GT/')"""

    config = parser.parse_args()
    print(config)
    main(config)