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
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBuild, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn(2, 1, self.hidden_size), requires_grad=True)  # need debug

    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MovementEncoderBuild(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementEncoderBuild, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        # need further improvement
        # input_temp = inputs.view(-1,8,17*13).permute(0, 2, 1)
        # input_spa = inputs.view(-1,8*17,13).permute(0, 2, 1)
        #
        # outputs_spa = self.main(input_spa).permute(0, 2, 1)
        # outputs_temp = self.main(input_temp).permute(0, 2, 1)
        #
        # outputs = torch.cat(outputs_spa, outputs_temp)
        inputs = inputs[:, :15, :]
        inputs = inputs.view(-1, 15, 13).permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)  # [1024,4,512]
        # print(outputs.shape)
        return self.out_net(outputs)


# ================================================= Evaluator =================================================
class MotionImgEvaluator():
    def __init__(self, motion_encoder, movement_encoder):
        self.device = "cuda:0"
        self.motion_encoder = motion_encoder
        self.movement_encoder = movement_encoder

        # model_dir = os.path.join("Anyskill/output/", '24354.tar')
        # checkpoints = torch.load(model_dir, map_location=self.device)
        # self.motion_encoder.load_state_dict(checkpoints['motion_encoder'])
        # self.movement_encoder.load_state_dict(checkpoints['movement_encoder'])
        # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoints['epoch']))

        self.motion_encoder.to(self.device)
        self.movement_encoder.to(self.device)

        self.motion_encoder.eval()
        self.movement_encoder.eval()

    def get_motion_embedding(self, motion):
        with torch.no_grad():
            m_lens = torch.ones(motion.shape[0])
            motion = motion.detach().to(self.device).float()  # [1,16,13] [1024,17,13]
            movements = self.movement_encoder(motion).detach()  # [1,4,512]
            # m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)  # [1,512]
        return motion_embedding


class TextToFeature:
    def __init__(self):
        self.mlip_model, _, self.mlip_preprocess = open_clip.create_model_and_transforms('ViT-B-32',
                                                                                         pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def encode_texts(self, texts):
        texts_token = self.tokenizer(texts)
        text_features = self.mlip_model.encode_text(texts_token).cuda()
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features_norm
