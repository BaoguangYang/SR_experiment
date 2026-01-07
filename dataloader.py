import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import numpy as np
import glob
import json
import configs

class Data(Dataset):
    def __init__(self, phase="train"):

        self.dataset = configs.DATASET if phase!= "test" else configs.TEST_DATASET
        self.seg_num = configs.SEG_NUM if phase!= "test" else configs.TEST_SEG_NUM
        self.frame_num = configs.FRAME_NUM if phase!= "test" else configs.TEST_FRAME_NUM
        self.seq_len = configs.SEQ_LEN
        self.H = configs.H
        self.W = configs.W
        self.h = configs.h
        self.w = configs.w

        self.train_seg_num = (self.seg_num * configs.TRAIN_PCT).astype(int)
        self.val_seg_num = self.seg_num - self.train_seg_num
        self.phase = phase

        curr_folder = os.path.abspath(os.curdir)

        self.color_lr = []
        self.depth_lr = []
        self.mv_lr = []
        self.jitter_lr = []
        self.color_hr = []
        # self.color_hr_native = []


        for i, data in enumerate(self.dataset):

            start = 0 if phase != "val" else self.train_seg_num[i]
            end = self.train_seg_num[i] if phase == "train" else self.seg_num[i]
            for seg in range(start, end):
                if phase != "test":
                    lr_path_input = curr_folder + "/data/QRISP/" + data + '/' + str(configs.h) + 'p/'
                    hr_path_input = curr_folder + "/data/QRISP/" + data + '/' + str(configs.H) + 'p/'
                else:
                    lr_path_input = curr_folder + "/data/QRISP/TestSet/" + data + '/' + str(configs.h) + 'p/'
                    hr_path_input = curr_folder + "/data/QRISP/TestSet/" + data + '/' + str(configs.H) + 'p/'

                if configs.jittering:
                    self.color_lr.append(lr_path_input + 'MipBiasMinus' + str(configs.MIP_BIAS) + 'Jittered/' + str(seg).zfill(4) + '/')
                    self.depth_lr.append(lr_path_input + 'DepthMipBiasMinus' + str(configs.MIP_BIAS) + 'Jittered/' + str(seg).zfill(4) + '/')
                    self.mv_lr.append(lr_path_input + 'MotionVectorsMipBiasMinus' + str(configs.MIP_BIAS) + 'Jittered/' + str(seg).zfill(4) + '/')
                    self.jitter_lr.append(lr_path_input + 'CameraData/' + str(seg).zfill(4) + '/')
                    if configs.MIP_BIAS_HR:
                        self.color_hr.append(hr_path_input + 'MipBiasMinus' + str(configs.MIP_BIAS_HR) + 'Jittered/' + str(seg).zfill(4) + '/')
                    else:
                        self.color_hr.append(hr_path_input + 'Enhanced/' + str(seg).zfill(4) + '/')
                else:
                    self.color_lr.append(lr_path_input + 'MipBiasMinus' + str(configs.MIP_BIAS) + '/' + str(seg).zfill(4) + '/')
                    self.depth_lr.append(lr_path_input + 'DepthMipBiasMinus' + str(configs.MIP_BIAS) + '/' + str(seg).zfill(4) + '/')
                    self.mv_lr.append(lr_path_input + 'MotionVectorsMipBiasMinus' + str(configs.MIP_BIAS) + '/' + str(seg).zfill(4) + '/')
                    self.jitter_lr.append(lr_path_input + 'CameraData/' + str(seg).zfill(4) + '/')
                    if configs.MIP_BIAS_HR:
                        self.color_hr.append(hr_path_input + 'MipBiasMinus' + str(configs.MIP_BIAS_HR) + '/' + str(seg).zfill(4) + '/')
                    else:
                        self.color_hr.append(hr_path_input + 'Enhanced/' + str(seg).zfill(4) + '/')

    def __len__(self):
        if self.phase == "train":
            seg_num = self.train_seg_num
        elif self.phase == "val":
            seg_num = self.val_seg_num
        elif self.phase == "test":
            return (np.sum(self.seg_num) * ((self.frame_num - 1) // self.seq_len))

        return (np.sum(seg_num) * (self.frame_num - self.seq_len)) # start from index 1


    def __getitem__(self, idx):
        # convert idx
        if self.phase != "test":
            frame_index = 1 + idx % (self.frame_num - self.seq_len) # start from index 1
            seg_index = idx // (self.frame_num - self.seq_len)
        else:
            frame_index = 1 + (idx % ((self.frame_num - 1) // self.seq_len)) * self.seq_len # start from index 1
            seg_index = idx // ((self.frame_num - 1) // self.seq_len)

        # [L, C, H, W]
        sample = {  'color_lr': torch.zeros((self.seq_len, 3, self.h, self.w)),
                    'depth_lr': torch.zeros((self.seq_len, 1, self.h, self.w)),
                    'mv_lr': torch.zeros((self.seq_len, 2, self.h, self.w)),
                    'jitter_lr': torch.zeros((self.seq_len, 2, 1, 1)),
                    'color_hr': torch.zeros((self.seq_len, 3, self.H, self.W)),
                    'prev_color_hr': torch.zeros((3, self.H, self.W)),
                    'prev_jitter': torch.zeros((2, 1, 1))
                }

        for i in range(self.seq_len):

            color_lr_path = self.color_lr[seg_index] + str(frame_index + i).zfill(4) + '.png'
            depth_lr_path = self.depth_lr[seg_index] + str(frame_index + i).zfill(4) + '.png'
            mv_lr_path = self.mv_lr[seg_index] + str(frame_index + i).zfill(4) + '.exr'
            jitter_lr_path = self.jitter_lr[seg_index] + str(frame_index + i).zfill(4) + '.json'
            color_hr_path = self.color_hr[seg_index] + str(frame_index + i).zfill(4) + '.png'

            color_lr = cv.cvtColor(cv.imread(color_lr_path)[:, :, :3], cv.COLOR_BGR2RGB) / 255
            depth_lr = cv.cvtColor(cv.imread(depth_lr_path, cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA)
            depth_lr = depth_lr[:, :, 0] / 255 + depth_lr[:, :, 1] / (255**2) + depth_lr[:, :, 2] / (255**3) + depth_lr[:, :, 3] / (255**4)
            
            mv_lr = cv.imread(mv_lr_path, cv.IMREAD_UNCHANGED)
            mv_lr = cv.cvtColor(mv_lr, cv.COLOR_BGR2RGB)[:, :, :2]

            sample['color_lr'][i] = torch.tensor(color_lr.transpose([2, 0, 1]))
            sample['depth_lr'][i] = torch.tensor(np.expand_dims(depth_lr, axis=0))
            sample['mv_lr'][i] = torch.tensor(mv_lr.transpose([2, 0, 1]))

            f = open(jitter_lr_path)
            json_lr = json.load(f)['jitter_offset']
            sample['jitter_lr'][i] = torch.tensor(np.expand_dims(np.array([json_lr['x'], json_lr['y']]), axis=(1,2)))
            f.close()

            color_hr = cv.cvtColor(cv.imread(color_hr_path)[:, :, :3], cv.COLOR_BGR2RGB) / 255
            sample['color_hr'][i] = torch.tensor(color_hr.transpose([2, 0, 1]))

        prev_color_hr_path = self.color_hr[seg_index] + str(frame_index - 1).zfill(4) + '.png'
        prev_color_hr = cv.cvtColor(cv.imread(prev_color_hr_path)[:, :, :3], cv.COLOR_BGR2RGB) / 255
        sample['prev_color_hr'] = torch.tensor(prev_color_hr.transpose([2, 0, 1])).to(torch.float32)

        prev_jitter_path = self.jitter_lr[seg_index] + str(frame_index - 1).zfill(4) + '.json'
        f = open(prev_jitter_path)
        json_lr = json.load(f)['jitter_offset']
        sample['prev_jitter'] = torch.tensor(np.expand_dims(np.array([json_lr['x'], json_lr['y']]), axis=(1,2)))
        f.close()

        sample["frame_index"] = frame_index
        sample["seg_index"] = seg_index

        return sample 


class Loader:
    def __init__(self, phase):
        self.dataset = Data(phase=phase)
        if phase == "train":
            self.batch_size = configs.BATCH_SIZE
            self.shuffle = True
        elif phase == "val":
            self.batch_size = configs.VAL_BATCH_SIZE
            self.shuffle = True
        elif phase == "test":
            self.batch_size = 1
            self.shuffle = False

        self.num_workers = 0
        self.dataloader = DataLoader(dataset=self.dataset,
                                    batch_size=self.batch_size,
                                    shuffle=self.shuffle,
                                    num_workers=self.num_workers,
                                    pin_memory=True) 