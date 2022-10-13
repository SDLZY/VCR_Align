import h5py
import os
from tqdm import tqdm
import numpy as np
import pandas as pd


data_dir = 'output/base_32000steps/ckpt/'
logitorprob = 'scores_softmax'  # scores/probs/scores_softmax
sim_fun = {
    'cos': lambda x, y: (x*y).sum(-1)/(np.linalg.norm(x, 2, -1) * np.linalg.norm(y, 2, -1)),
    'mse': lambda x, y: np.square(x-y).mean(-1),
    'l1': lambda x, y: np.linalg.norm(x-y, 1, -1),
    'l2': lambda x, y: np.linalg.norm(x-y, 2, -1),
}


def get_att(data, choice):
    mask = data[f'mask{choice}'][:] > 0
    att = data[f'attention{choice}'][:][:, :, mask]
    if logitorprob == 'scores_softmax':
        att = np.exp(att)/np.sum(np.exp(att), keepdims=True, axis=-1)
    return att


if __name__ == '__main__':
    if logitorprob == 'scores_softmax':
        hp = h5py.File(os.path.join(data_dir, f'attention_scores.h5'), 'r')
    else:
        hp = h5py.File(os.path.join(data_dir, f'attention_{logitorprob}.h5'), 'r')
    # hp_new = h5py.File(os.path.join(data_dir, f'attention_{logitorprob}_stats.h5'), 'w')
    df = []
    for anno_id, data in tqdm(hp.items(), total=len(hp), ncols=80):
        mode, idx = anno_id.split('-')
        idx = int(idx)
        # if idx > 100:
        #     break
        data_ans = data['q2a']
        data_rat = data['qa2r']

        target_ans = np.array(data_ans['target']).item()
        target_rat = np.array(data_rat['target']).item()

        att_ans_gt = get_att(data_ans, target_ans)
        att_rat_gt = get_att(data_rat, target_rat)

        for choice in range(4):
            att_ans = get_att(data_ans, choice)
            att_rat = get_att(data_rat, choice)
            # item_ans = {'idx': anno_id, 'anchor_task': 'q2a', 'anchor_choice': target_ans, 'task': 'qa2r', 'choice': choice}
            # item_rat = {'idx': anno_id, 'anchor_task': 'qa2r', 'anchor_choice': target_rat, 'task': 'q2a', 'choice': choice}
            for layer in range(8,12):
                for head in range(12):
                    item_ans = {'idx': anno_id, 'anchor_task': 'q2a', 'choice': choice, 'layer': layer, 'head': head,
                                'pos': choice == target_rat}
                    item_rat = {'idx': anno_id, 'anchor_task': 'qa2r', 'choice': choice, 'layer': layer, 'head': head,
                                'pos': choice == target_ans}
                    for metric, fun in sim_fun.items():
                        # import pdb; pdb.set_trace()
                        sim2ans = fun(att_ans_gt[layer, head], att_rat[layer, head]).item()
                        sim2rat = fun(att_rat_gt[layer, head], att_ans[layer, head]).item()
                        item_ans[metric] = sim2ans
                        item_rat[metric] = sim2rat
                    df.append(item_ans)
                    df.append(item_rat)
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(data_dir, f'attention_{logitorprob}_stat_raw.csv'))
    df_avg = df.groupby(['anchor_task', 'layer', 'head', 'pos'])['cos', 'mse', 'l1', 'l2'].mean()
    df_avg.to_csv(os.path.join(f'attention_{logitorprob}_stat.csv'))
    print(df_avg)

