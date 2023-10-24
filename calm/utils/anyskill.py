import wandb
import time
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from Anyskill.network.evaluator import *
from Anyskill.dataset.dataset import ASDataset
# from Anyskill.utils.parser import test_options
from Anyskill.utils.utils import *

def evaluation(img_emb, motion_emb, matching_score_sum, top_k_count, all_size):
    dist_mat = euclidean_distance_matrix(img_emb.cpu().numpy(), motion_emb.cpu().numpy())
    matching_score_sum += dist_mat.trace()
    argsmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argsmax, top_k=1)
    top_k_count += top_k_mat.sum(axis=0)

    all_size += img_emb.shape[0]
    matching_score = matching_score_sum / all_size
    R_precision = top_k_count / all_size

    return matching_score, R_precision

def anytest():
    # args = test_options()

    motion_encoder = MotionEncoderBuild(
        hidden_size=512,
        output_size=512,
        device="cuda:0"
    )

    evaluator = MotionImgEvaluator(motion_encoder)

    return evaluator

if __name__ == "__main__":
    evaluator = anytest()
    test_dataset = ASDataset(args.test_motion_file, args.test_image_file)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                              shuffle=True)
    output = {}
    matching_score_sum = 0
    top_k_count = 0
    all_size = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            motion, img_emb, n_motions, n_embs, m_lens, m_id = data
            id = m_id.item()
            pred_motion = evaluator.get_motion_embedding(motion, m_lens)
            # print("pred_motion is: ", pred_motion.shape) # [1, 512](batch_size)
            output[id] = pred_motion.data.cpu().numpy()

            matching_score, R_precision = evaluation(img_emb.squeeze(0), pred_motion, matching_score_sum, top_k_count, all_size)
            output["score"] = matching_score
            output["R_prc"] = R_precision
            print(f'---> {id}th M2I pairs\' Matching Score: {matching_score:.4f}')

            # line = f'---> [{m_id}] R_precision: '
            # for i in range(len(R_precision)):
            #     line += '(top %d): %.4f ' % (i+1, R_precision[i])
            # print(line)

    # save the result
    # np.save("", output)


