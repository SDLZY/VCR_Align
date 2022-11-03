import h5py
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb


data_dir = 'output/align_new/base_pat5_nstep12000_align/l1_probs_neg_mu0.1_alpha0.1/ckpt/'


def get_att(data, choice):
    mask = data[f'mask{choice}'][:] > 0
    att = data[f'attention{choice}'][:][:, :, mask]
    return att


if __name__ == '__main__':
    hp = h5py.File(os.path.join(data_dir, f'attention_probs.h5'), 'r')
    # hp_new = h5py.File(os.path.join(data_dir, f'attention_{logitorprob}_stats.h5'), 'w')
    results = []
    for idx in tqdm(range(26534)):
    # for idx in tqdm(range(10)):
        data = hp[f'val-{idx}']

        data_qa = data['q2a']
        data_qar = data['qa2r']

        for choice in range(4):
            att_qa = data_qa[f'attention{choice}'][:]
            ids_qa = data_qa[f'input_ids{choice}'][:]

            att_qar = data_qar[f'attention{choice}'][:]
            ids_qar = data_qar[f'input_ids{choice}'][:]

            num_layers = att_qa.shape[0]
            # print(num_layers)
            # print(att_qa.sum(-1))
            assert (np.abs(att_qa.sum(-1) - 1.) < 0.01).all()
            assert (np.abs(att_qar.sum(-1) - 1.) < 0.01).all()

            sep_pos_qa = np.where(ids_qa == 102)[0].tolist()
            sep_pos_qar = np.where(ids_qar == 102)[0].tolist()

            # import pdb; pdb.set_trace()
            slices = [slice(i + 1, j) for i, j in zip([-1] + sep_pos_qar[:-1], sep_pos_qar)]
            att_map = np.empty((*att_qar.shape[:2], 3, 3))
            for i, qidx in enumerate(slices):
                for j, kidx in enumerate(slices):
                    att_map[:, :, i, j] = att_qar[:, :, qidx, kidx].sum(-1).mean(-1)
            # print(att_map.shape)
            results.append(att_map)

    results = np.array(results)
    pdb.set_trace()
    print(results.shape)
    # print(results)
    np.save(os.path.join(data_dir, 'qav_att_map.npy'), results)