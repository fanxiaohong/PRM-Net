import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.transforms as transform

CROP_SIZE = 32  # x/y裁剪尺寸
CROP_SIZE_z = 32  # z裁剪尺寸
################################################################################################3
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, ids_patinet: list, aug_mode):
        self.images_path = images_path
        self.images_class = images_class
        self.ids_patinet = ids_patinet
        self.aug_mode = aug_mode

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):

        # 读取图像和mask数据并归一化
        str_list = self.images_path[item]
        str_split = str_list.split('/')
        image_str_t1 = self.images_path[item] + '/' + str_split[3] + 'A1.nii.gz'
        image_str_t2 = self.images_path[item] + '/' + str_split[3] + 'A0.nii.gz'
        image_str_t3 = self.images_path[item] + '/' + str_split[3] + 'T2.nii.gz'
        mask_str_t1 = self.images_path[item] + '/se' + str_split[3] + 'A1.nii.gz'

        def get_brain_region(image, mask):
            indice_list = np.where(mask > 0)
            # calculate the min and max of the indice,  here volume have 3 channels
            channel_0_min = min(indice_list[0])
            channel_0_max = max(indice_list[0])

            channel_1_min = min(indice_list[1])
            channel_1_max = max(indice_list[1])

            channel_2_min = min(indice_list[2])
            channel_2_max = max(indice_list[2])

            # center crop
            c_center = (channel_0_min + channel_0_max) // 2
            h_center = (channel_1_min + channel_1_max) // 2
            w_center = (channel_2_min + channel_2_max) // 2

            brain_volume = image[c_center - CROP_SIZE_z // 2:c_center + CROP_SIZE_z // 2,
                           h_center - CROP_SIZE // 2:h_center + CROP_SIZE // 2,
                           w_center - CROP_SIZE // 2:w_center + CROP_SIZE // 2]

            # broad 5 pixels
            return brain_volume

        def img_mask(image_str, mask_str, trunc_min, trunc_max):
            img = sitk.ReadImage(image_str, sitk.sitkFloat32)
            img = sitk.GetArrayFromImage(img)
            img = torch.tensor(img, dtype=torch.float32)
            img[img <= trunc_min] = trunc_min  # 下限截断
            img = (img - trunc_min) / (img.max() - trunc_min)
            mask = sitk.ReadImage(mask_str, sitk.sitkFloat32)
            mask = sitk.GetArrayFromImage(mask)
            mask = torch.tensor(mask, dtype=torch.float32)
            img = img * mask

            # 将脑部抠出
            out_img = get_brain_region(img, mask)
            img = torch.unsqueeze(torch.from_numpy(np.array(out_img)), dim=0)
            return img

        # 拼装四种img
        image_t1 = img_mask(image_str_t1, mask_str_t1, 0, 600)  # A1=[0,300]
        image_t2 = img_mask(image_str_t2, mask_str_t1, 0, 600)  # A0=[0,300]
        image_t3 = img_mask(image_str_t3, mask_str_t1, 0, 600)  # T2=[0,300]
        img = torch.cat((image_t1, image_t2, image_t3), 0)
        img = img.cpu().data.numpy()

        # data augment
        if self.aug_mode == 1:
            if random.random() < 0.5:  # random_horizontal_flip
                img = img[:, :, :, ::-1].copy()

            if random.random() < 0.5:  # random_vertical_flip
                img = img[:, :, ::-1, :].copy()

            if random.random() < 0.5:  # random_vertical_flip
                img = img[:, ::-1, :, :].copy()

        img = torch.from_numpy(img)
        label = self.images_class[item]
        id_patient = self.ids_patinet[item]
        return img, label, id_patient

    @staticmethod
    def collate_fn(batch):
        images, labels, ids_patient = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        ids_patient = torch.as_tensor(ids_patient)
        return images, labels, ids_patient

