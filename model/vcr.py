"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VCR model
"""
from collections import defaultdict

from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

# from .layer import GELU
from .model import (
    UniterPreTrainedModel, UniterModel)


class UniterForVisualCommonsenseReasoning(UniterPreTrainedModel):
    """ Finetune UNITER for VCR
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)
        self.vcr_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.ReLU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, 2)
        )
        self.apply(self.init_weights)

    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.uniter.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embeddings.word_embeddings = new_emb

    def forward(self, batch, compute_loss=True, output_attention=False, logitorprob='probs', draw_attention=None):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_type_ids = batch['txt_type_ids']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      txt_type_ids=txt_type_ids,
                                      output_attention=output_attention, logitorprob=logitorprob, draw_attention=draw_attention)
        if output_attention:
            sequence_output, cls_att_dict, box_mask = sequence_output
        else:
            sequence_output = sequence_output

        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.vcr_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            vcr_loss = F.cross_entropy(
                    rank_scores, targets.squeeze(-1),
                    reduction='mean')
            if output_attention:
                return vcr_loss, rank_scores[:, 1:], cls_att_dict, box_mask
            return vcr_loss
        else:
            rank_scores = rank_scores[:, 1:]
            if output_attention:
                return rank_scores, cls_att_dict, box_mask
            return rank_scores
