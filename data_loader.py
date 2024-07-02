import glob
import os
import numpy as np
import  csv
import pandas as pd
########################################################################
def split_data(traindata_str, testdata_str,root):
    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    train_id = []
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    val_id = []
    ######################################
    print('训练中数据统计：')
    for i in np.unique(traindata_str['label'].values):
        print('num of '+str(i)+'=', np.sum(traindata_str['label'].values == i))  # 对照unique数组，依次统计每个元素出现的次数

    print('验证中数据统计：')
    for i in np.unique(testdata_str['label'].values):
        print('num of '+str(i)+'=', np.sum(testdata_str['label'].values == i))  # 对照unique数组，依次统计每个元素出现的次数
    ######################################
    for i in range(len(traindata_str['id'])):
        label_tmp = traindata_str['label'].values[i]
        id_tmp = traindata_str['id'].values[i]
        root_tmp = root + str(label_tmp)+ '/'+str(traindata_str['id'].values[i])
        train_images_path.append(root_tmp)  # 存储训练集的所有图片路径
        train_images_label.append(int(label_tmp))  # 存储训练集label对应索引信息
        train_id.append(int(id_tmp))  # 存储训练集id对应索引信息

    for i in range(len(testdata_str['id'])):
        label_tmp = testdata_str['label'].values[i]
        id_tmp = testdata_str['id'].values[i]
        root_tmp = root + str(label_tmp) + '/' + str(testdata_str['id'].values[i])
        val_images_path.append(root_tmp)  # 存储验证集的所有图片路径
        val_images_label.append(int(label_tmp))  # 存储验证集图片对应索引信息
        val_id.append(int(id_tmp))  # 存储训练集id对应索引信息

    return train_images_path, train_images_label,train_id, val_images_path, val_images_label,val_id


