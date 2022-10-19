import h5py
import os
from tqdm import tqdm
import numpy as np
import pandas as pd


data_dir = 'output/align_new/base_pat5_nstep8000_align/alpha0.1_gamma5/ckpt/'


def get_att(data, choice):
    mask = data[f'mask{choice}'][:] > 0
    att = data[f'attention{choice}'][:][:, :, mask]
    return att


if __name__ == '__main__':
    hp = h5py.File(os.path.join(data_dir, f'attention_probs.h5'), 'r')
    # hp_new = h5py.File(os.path.join(data_dir, f'attention_{logitorprob}_stats.h5'), 'w')
    for task in ('q2a', 'qa2r'):
        results_map = []
        results_cls = []
        # for idx in tqdm(range(10)):
        for idx in tqdm(range(26534)):
            data = hp[f'val-{idx}'][task]

            for choice in range(4):
                att = data[f'attention{choice}'][:]
                ids = data[f'input_ids{choice}'][:]

                tv_split = np.where(ids == 102)[0].tolist()[-1] + 1
                # import pdb; pdb.set_trace()
                att_map = np.empty((*att.shape[:2], 2, 2))
                att_cls = np.empty((*att.shape[:2], 2))
                att_map[:, :, 0, 0] = att[:, :, :tv_split, :tv_split].sum(-1).mean(-1)
                att_map[:, :, 0, 1] = att[:, :, :tv_split, tv_split:].sum(-1).mean(-1)
                att_map[:, :, 1, 0] = att[:, :, tv_split:, :tv_split].sum(-1).mean(-1)
                att_map[:, :, 1, 1] = att[:, :, tv_split:, tv_split:].sum(-1).mean(-1)
                att_cls[:, :, 0] = att[:, :, 0, :tv_split].sum(-1)
                att_cls[:, :, 1] = att[:, :, 0, tv_split:].sum(-1)

                results_map.append(att_map)
                results_cls.append(att_cls)
        results_map = np.array(results_map)
        results_cls = np.array(results_cls)
        print(f'TASK: {task}')
        print(results_map.mean((0, 1, 2)))
        print(results_cls.mean((0, 1, 2)))
        np.save(os.path.join(os.path.dirname(data_dir), f'tv_att_map_{task}.npy'), results_map)
        np.save(os.path.join(os.path.dirname(data_dir), f'cls2tv_att_{task}.npy'), results_cls)