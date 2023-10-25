import os
import time
# import wandb
import torch
import codecs
# import transformers
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from Anyskill.utils.utils import *
import open_clip


def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MotionEncoderBuild(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(MotionEncoderBuild, self).__init__()
        self.device = device
        self.main = nn.Sequential(
            nn.Flatten(),  # Flatten the input to [1, 15*3]
            nn.Linear(15 * 3, hidden_size),  # Input layer: 15*3 input features, 256 output features
            nn.ReLU(),  # ReLU activation function
            nn.Linear(hidden_size, output_size),  # Hidden layer: 256 input features, 512 output features
            # nn.ReLU(),  # ReLU activation function
            # nn.Linear(hidden_size, output_size)  # Output layer: 512 input features, 512 output features
        )

        self.main.apply(init_weight)

    def forward(self, inputs):
        num_samples = inputs.shape[0]
        inputs = inputs.view(-1, 15, 3) #[32,15,3]
        output = self.main(inputs)
        return output


# ================================================= Evaluator =================================================
class MotionImgEvaluator():
    def __init__(self, motion_encoder):
        self.device = "cuda:0"
        self.motion_encoder = motion_encoder

        model_dir = "./Anyskill/output/finest.tar"
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.motion_encoder.load_state_dict(checkpoints['motion_encoder'])
        print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoints['epoch']))

        self.motion_encoder.to(self.device)

        self.motion_encoder.eval()

    def get_motion_embedding(self, motion):
        with torch.no_grad():
            m_lens = torch.ones(motion.shape[0])
            motion = motion.detach().to(self.device).float()  # [1,16,13] [1024,17,13]
            motion_embedding = self.motion_encoder(motion)  # [1,4,512]
        return motion_embedding


class FeatureExtractor():
    def __init__(self):
        self.mlip_model, _, self.mlip_preprocess = open_clip.create_model_and_transforms('ViT-B-32',
                                                                                         pretrained='laion2b_s34b_b79k', device="cuda")
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def encode_texts(self, texts):
        texts_token = self.tokenizer(texts).cuda()
        text_features = self.mlip_model.encode_text(texts_token).cuda()
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features_norm

    def encode_images(self, images):
        return self.mlip_model.encode_image(images)