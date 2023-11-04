import numpy as np
import torch
from torch.utils import data
import os
from tqdm import tqdm, trange


class ASDataset(data.Dataset):
    def __init__(self, motion_file, image_file):
        self.pointer = 0
        self.window_length = 8
        raw_motions = np.load(motion_file, allow_pickle=True) #motion[n,17,13]
        raw_emb = np.load(image_file, allow_pickle=True) #img[n,1,512]
        max_length_motion = raw_motions.shape[0]
        max_length_emb = raw_emb.shape[0]
        assert max_length_motion == max_length_emb, "The length of motion and clip feature are not the same"
        if max_length_motion % 16:
            keep = max_length_motion - (max_length_motion % 16)
            raw_motions = raw_motions[:keep, :, :]
            raw_emb = raw_emb[:keep, :, :]
        else:
            keep = max_length_motion
        data_dict = {}
        name_list = []

        for item in trange(keep-8):
            try:
                name_list.append(item)
                motion = raw_motions[item, :15, :3]
                img_emb = raw_emb[item, :, :]
                n_motions = raw_motions[item: item + self.window_length, :15, :3]
                # print(n_motions.shape[0])
                n_embs = raw_emb[item: item + self.window_length, :, :]
                data_dict[item] = {
                    'motion': motion,
                    'img_emb': img_emb,
                    'n_motions': n_motions,
                    'n_embs': n_embs,
                    'length': 1,
                    'm_id':item
                    # 'length': self.window_length
                }
            except:
                pass
        self.data_dict = data_dict
        self.name_list = name_list
        self.length_arr = max_length_motion
        # self.mean = mean
        # self.std = std

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        motion = self.data_dict[item]['motion'] #[16,3]
        img_emb = self.data_dict[item]['img_emb'] #[1,512]
        n_motions = self.data_dict[item]['n_motions'] #[8,16,3]
        n_embs = self.data_dict[item]['n_embs'].squeeze() #[8,512]
        m_len = self.data_dict[item]['length']
        m_id = self.data_dict[item]['m_id']

        # # Normalization
        # motion = (motion - self.mean) / self.std

        return motion, img_emb, n_motions, n_embs, m_len, m_id
