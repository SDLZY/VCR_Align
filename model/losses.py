import math
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F


class InnerProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask1: torch.Tensor, y: torch.Tensor, mask2: torch.Tensor, label=None):
        x = x * mask1.float()
        y = y * mask2.float()
        z = torch.matmul(x, y.transpose(-1, -2))
        return z


class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1: torch.Tensor, mask1: torch.Tensor,
                input2: torch.Tensor, mask2: torch.Tensor, label=None):
        input1 = input1 * mask1.float()
        input2 = input2 * mask2.float()
        input1 = F.normalize(input1, dim=-1)
        input2 = F.normalize(input2, dim=-1)
        cosine = torch.matmul(input1, input2.transpose(-1, -2))
        # z = z / (x.norm(2, -1, keepdim=True) * y.norm(2, -1, keepdim=True))
        return cosine


class NegativeMSE(nn.Module):
    """mean square error"""
    def __init__(self):
        super(NegativeMSE, self).__init__()

    def forward(self, input1: torch.Tensor, mask1: torch.Tensor,
                input2: torch.Tensor, mask2: torch.Tensor, label=None):
        """input shape: (bs, 4/1, num_head*n_obj)"""
        input1 = input1 * mask1.float()
        input2 = input2 * mask2.float()
        dist = ((input1.unsqueeze(2) - input2.unsqueeze(1))**2).sum(-1)  # (bs, 4/1, 4/1)
        mask = (mask1.unsqueeze(2) * mask2.unsqueeze(1)).sum(-1).float()
        return -dist/mask


class NegativeSSE(nn.Module):
    """sum square error"""
    def __init__(self):
        super(NegativeSSE, self).__init__()

    def forward(self, input1: torch.Tensor, mask1: torch.Tensor,
                input2: torch.Tensor, mask2: torch.Tensor, label=None):
        """input shape: (bs, 4/1, num_head*n_obj)"""
        input1 = input1 * mask1.float()
        input2 = input2 * mask2.float()
        dist = ((input1.unsqueeze(2) - input2.unsqueeze(1))**2).sum(-1)  # (bs, 4/1, 4/1)
        return -dist


class NegativeKLDiv(nn.Module):
    def __init__(self):
        super().__init__()

    def _kl_div(self, p, q):
        eps = 1e-9
        return (p * ((p + eps).log() - (q + eps).log())).sum(-1)

    def forward(self, input1: torch.Tensor, mask1: torch.Tensor,
                input2: torch.Tensor, mask2: torch.Tensor, label=None):
        """input shape: (bs, 4/1, num_head*n_obj)"""
        input1 = input1 * mask1.float()
        input2 = input2 * mask2.float()
        div = self._kl_div(input1.unsqueeze(2), input2.unsqueeze(1))
        return -div


class AddMarginProduct(nn.Module):
    r"""cosface"""

    def __init__(self, s=1.0, m=0.0):
        super(AddMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self._cos = CosineSimilarity()

    def forward(self, input1, mask1, input2, mask2, label):
        bs = label.shape[0]
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = self._cos(input1, mask1, input2, mask2).view(bs, -1)
        # input1 = F.normalize(input1, dim=-1)
        # input2 = F.normalize(input2, dim=-1)
        # cosine = torch.matmul(input1, input2.transpose(-1, -2)).view(bs, -1)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ArcMarginProduct(nn.Module):
    r"""arcface"""

    def __init__(self, s=1.0, m=0.0, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self._cos = CosineSimilarity()

    def forward(self, input1, mask1, input2, mask2, label):
        bs = label.shape[0]
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # input1 = F.normalize(input1, dim=-1)
        # input2 = F.normalize(input2, dim=-1)
        # cosine = torch.matmul(input1, input2.transpose(-1, -2)).view(bs, -1)
        cosine = self._cos(input1, mask1, input2, mask2).view(bs, -1)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class AppRank(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, input1, mask1, input2, mask2, label=None):
        """input1 shape (bs, 4, head, dim), input2 shape (bs, 1, head, dim)"""
        bs, _, num_heads, _ = input1.shape
        input1 = input1.permute((0, 2, 1, 3)).contiguous().view(bs*num_heads, 4, -1)
        mask1 = mask1.permute((0, 2, 1, 3)).contiguous().view(bs*num_heads, 4, -1)
        input2 = input2.permute((0, 2, 1, 3)).contiguous().view(bs*num_heads, 1, -1)
        mask2 = mask2.permute((0, 2, 1, 3)).contiguous().view(bs*num_heads, 1, -1)

        input1 = input1 * mask1.float()
        input2 = (input2 * mask2.float()).detach()

        diff = input1.unsqueeze(-1) - input1.unsqueeze(-2)
        diff = torch.sigmoid(-diff * self.alpha)  # b, 4, n, n

        mask1_ = mask1.unsqueeze(-1) * mask1.unsqueeze(-2)
        pi = 0.5 + (diff * mask1_.float()).sum(-1)  # b, 4, n
        app_dcg = input2.exp() / (1 + pi).log()
        app_dcg = (app_dcg * mask1.float()).sum(-1)

        input2 = torch.where(mask2 > 0, input2, input2.new_full((1,), -1E5))
        input2_sort, input2_sorted_indices = input2.sort(-1, descending=True)
        mask2 = torch.gather(mask2, dim=-1, index=input2_sorted_indices)
        idcg = input2_sort.exp() / torch.arange(2, input2.shape[-1]+2, device=input2.device).float().log()
        idcg = (idcg * mask2.float()).sum(-1)
        ndcg = app_dcg / idcg
        ndcg = ndcg.view(bs, num_heads, -1)
        return ndcg


class ListMLE(nn.Module):
    def __init__(self, alpha=1.0, mode='exp'):
        super().__init__()
        self.alpha = alpha
        assert mode in ('exp', 'linear', 'sigmoid')
        self.mode = mode

    def forward(self, input1, mask1, input2, mask2, label=None):
        """shape (bs, 4, num_heads, n)"""
        input2 = torch.where(mask2 > 0, input2, input2.new_full((1,), 1E5))
        targets_sorted_indices = input2.sort(-1)[1]  # 升序
        mask_sorted = torch.gather(mask2, dim=-1, index=targets_sorted_indices)

        inputs_sorted = torch.gather(input1, dim=-1, index=targets_sorted_indices.repeat(1, 4, 1, 1))
        if self.mode == 'exp':
            inputs_sorted_proj = torch.exp(self.alpha * inputs_sorted) * mask_sorted.float()
        elif self.mode == 'linear':
            inputs_sorted_proj = inputs_sorted
        elif self.mode == 'sigmoid':
            inputs_sorted_proj = torch.sigmoid(self.alpha * inputs_sorted) * mask_sorted.float()
        else:
            raise ValueError

        eps = 1e-9
        inputs_sorted_proj_cum = inputs_sorted_proj.cumsum(dim=-1) * mask_sorted.float()
        # 连乘导致结果数量级太小
        inputs_prob = torch.prod((inputs_sorted_proj + eps) / (inputs_sorted_proj_cum + eps), dim=-1)  # bs, 4, num_head
        inputs_prob = inputs_prob.permute((0, 2, 1)).contiguous()
        return inputs_prob


class Spearman(nn.Module):
    def __init__(self, alpha=1.0, measure='l2'):
        super().__init__()
        self.alpha = alpha
        assert measure in ('l1', 'l2', 'smooth_l1')
        if measure == 'l1':
            self._measure = nn.L1Loss(reduction='none')
        elif measure == 'l2':
            self._measure = nn.MSELoss(reduction='none')
        elif measure == 'smooth_l1':
            self._measure = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError

    def forward(self, input1, mask1, input2, mask2, label=None):
        """shape (bs, 4, num_heads, n)"""
        # 两次排序找出元素在降序（升序）中的位置 https://blog.csdn.net/LXX516/article/details/78804884
        input2 = torch.where(mask2 > 0, input2, input2.new_full((1,), -1E5))
        targets_sorted_indexes = input2.sort(dim=-1, descending=True)[1].sort(dim=-1)[1].float() + 1
        targets_sorted_indexes = targets_sorted_indexes * mask2.float()
        _mask1 = mask1.unsqueeze(-1) * mask1.unsqueeze(-2)
        # targets = targets.detach()
        # targets_dist = (targets.unsqueeze(-1) - targets.unsqueeze(-2))
        # targets_sorted_indexes = 0.5 + (torch.sigmoid(-targets_dist * self.alpha) * _mask.float()).sum(-1)
        inputs_dist = (input1.unsqueeze(-1) - input1.unsqueeze(-2))
        inputs_sorted_indexes = 0.5 + (torch.sigmoid(-inputs_dist * self.alpha) * _mask1.float()).sum(-1)
        inputs_sorted_indexes = inputs_sorted_indexes * mask1.float()
        # 需要保证padding部分数值为0且对measure函数无影响
        dist = self._measure(targets_sorted_indexes.expand_as(inputs_sorted_indexes), inputs_sorted_indexes)
        d = mask1.sum(-1)
        corr = 1 - 6 * dist.sum(-1)/(d*(d**2 - 1)).float()
        corr = corr.permute((0, 2, 1)).contiguous()
        return corr


class EmbeddingLoss(nn.Module):
    """最大化正例间的相似度，在margin基础上最小化负例间的相似度"""
    def __init__(self, margin=0.0):
        super(EmbeddingLoss, self).__init__()
        self.margin = margin

    def forward(self, predict, target):
        if len(predict.shape) == 2:
            predict = predict.unsqueeze(1)  # 无head时扩一维视作有一个head
        bs, num_heads, n_ex = predict.shape
        predict = predict.view(bs*num_heads, n_ex)
        target = target.unsqueeze(1).repeat(1, num_heads).view(-1)
        pred_pos = predict[torch.arange(bs * num_heads), target]  # bs*num_heads
        neg_mask = torch.ones_like(predict).to(torch.uint8)
        neg_mask[torch.arange(bs * num_heads), target] = 0
        pred_neg = predict[neg_mask > 0].view(bs * num_heads, n_ex-1)  # bs*num_heads, num_neg

        loss_pos = (1 - pred_pos).mean()
        loss_neg = F.relu(pred_neg - self.margin).sum(-1).mean()
        loss = loss_neg + loss_pos
        return loss


class MarginRankingLoss(nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self._loss = nn.MarginRankingLoss(margin=margin)
        # self._loss = nn.MarginRankingLoss(margin=margin, reduction='sum')

    def forward(self, predict, target):
        """predict shape (bs, head, n) n=4 or 16"""
        if len(predict.shape) == 2:
            predict = predict.unsqueeze(1)  # 无head时扩一维视作有一个head
        bs, num_heads, n_ex = predict.shape
        predict = predict.view(bs*num_heads, n_ex)
        target = target.unsqueeze(1).repeat(1, num_heads).view(-1)
        pred_pos = predict[torch.arange(bs * num_heads), target].unsqueeze(1).repeat(1, n_ex-1)
        neg_mask = torch.ones_like(predict).to(torch.uint8)
        neg_mask[torch.arange(bs * num_heads), target] = 0
        pred_neg = predict[neg_mask > 0].view(bs * num_heads, n_ex-1)
        ones = torch.ones_like(pred_pos)
        loss = self._loss(pred_pos, pred_neg, ones)
        return loss


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, predict, target):
        """predict shape (bs, n)"""
        one_hot = torch.zeros_like(predict).scatter(dim=1, index=target.unsqueeze(1).long(), value=1)
        bce = F.binary_cross_entropy_with_logits(predict, one_hot)
        return bce


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, predict, target):
        one_hot = torch.zeros_like(predict).scatter(dim=1, index=target.unsqueeze(1).long(), value=1)
        bce = F.binary_cross_entropy_with_logits(predict, one_hot, reduction='none')
        pt = torch.exp(-bce)
        loss = (1 - pt)**self.gamma * bce
        return loss.mean()


class AlignLoss(nn.Module):
    """2x4CE"""
    def __init__(
            self,
            sim_fn,
            loss_fn,
            layers=(11, ),
            per_head=False,
            head_average=False,
    ):
        super().__init__()
        self.sim_fn = sim_fn
        self.loss_fn = loss_fn
        self.layers = layers
        self.acc1 = dict((layer, 0.) for layer in layers)
        self.acc2 = dict((layer, 0.) for layer in layers)
        self.loss1 = dict((layer, 0.) for layer in layers)
        self.loss2 = dict((layer, 0.) for layer in layers)
        self.count = 0
        self.per_head = per_head
        self.head_average = head_average

    def forward(self, att1_list: Dict[int, torch.Tensor], att1_mask: torch.Tensor, target1: torch.Tensor,
                att2_list: Dict[int, torch.Tensor], att2_mask: torch.Tensor, target2: torch.Tensor, record=True):
        """att shape (bs, 4, head, n_obj); target is in ce format"""
        output_dict = {}
        loss = 0
        bs = att1_mask.shape[0]
        if not self.per_head and not self.head_average:
            att1_mask = att1_mask.view(bs, 4, -1)
            att2_mask = att2_mask.view(bs, 4, -1)
        elif not self.per_head and self.head_average:
            att1_mask = att1_mask[:, :, 0]
            att2_mask = att2_mask[:, :, 0]
        else:
            assert False
        for layer in self.layers:
            att1 = att1_list[layer]
            att2 = att2_list[layer]
            if not self.per_head and not self.head_average:
                att1 = att1.view(bs, 4, -1)  # head layer
                att2 = att2.view(bs, 4, -1)
            elif not self.per_head and self.head_average:
                att1 = att1.mean(dim=2)
                att2 = att2.mean(dim=2)
            else:
                assert False
            att1_gt = att1[torch.arange(bs), target1.long(), None]
            att2_gt = att2[torch.arange(bs), target2.long(), None]

            sim1 = self.sim_fn(att1, att1_mask, att2_gt, att2_mask[:, [0]], target1).squeeze(-1)
            sim2 = self.sim_fn(att2, att2_mask, att1_gt, att1_mask[:, [0]], target2).squeeze(-1)

            loss1 = self.loss_fn(sim1, target1)
            loss2 = self.loss_fn(sim2, target2)

            loss += (loss1 + loss2)/2
            output_dict[f'loss1_{layer}'] = loss1.item()
            output_dict[f'loss2_{layer}'] = loss2.item()
            if not self.per_head:
                output_dict[f'acc1_{layer}'] = (sim1.argmax(-1) == target1).float().mean().item()
                output_dict[f'acc2_{layer}'] = (sim2.argmax(-1) == target2).float().mean().item()
                output_dict['q2a_align_logits'] = sim1
                output_dict['qa2r_align_logits'] = sim2
            else:
                output_dict[f'acc1_{layer}'] = (sim1.argmax(-1) == target1[..., None]).float().mean().item()
                output_dict[f'acc2_{layer}'] = (sim2.argmax(-1) == target2[..., None]).float().mean().item()
            if record:
                self.loss1[layer] += output_dict[f'loss1_{layer}'] * bs
                self.loss2[layer] += output_dict[f'loss2_{layer}'] * bs
                self.acc1[layer] += output_dict[f'acc1_{layer}'] * bs
                self.acc2[layer] += output_dict[f'acc2_{layer}'] * bs
        if record:
            self.count += bs
        loss /= len(self.layers)
        return loss, output_dict

    def reset(self):
        self.count = 0
        for layer in self.layers:
            self.loss1[layer] = 0.
            self.loss2[layer] = 0.
            self.acc1[layer] = 0.
            self.acc2[layer] = 0.

    def get_metrics(self, reset: bool = False):
        metrics = {}
        if self.count > 0:
            for layer in self.layers:
                metrics[f'loss1_{layer}'] = self.loss1[layer]/self.count
                metrics[f'loss2_{layer}'] = self.loss2[layer]/self.count
                metrics[f'acc1_{layer}'] = self.acc1[layer]/self.count
                metrics[f'acc2_{layer}'] = self.acc2[layer]/self.count
        else:
            for layer in self.layers:
                metrics[f'loss1_{layer}'] = 0.
                metrics[f'loss2_{layer}'] = 0.
                metrics[f'acc1_{layer}'] = 0.
                metrics[f'acc2_{layer}'] = 0.
        if reset:
            self.reset()
        return metrics


class AlignLoss16(nn.Module):
    def __init__(
            self,
            sim_fn,
            loss_fn,
            layers=(11, ),
            per_head=False,
            head_average=False
    ):
        super().__init__()
        self.sim_fn = sim_fn
        self.loss_fn = loss_fn
        self.layers = layers
        self.acc = dict((layer, 0.) for layer in layers)
        self.loss = dict((layer, 0.) for layer in layers)
        self.count = 0
        self.per_head = per_head
        self.head_average = head_average

    def forward(self, att1_list: Dict[int, torch.Tensor], att1_mask: torch.Tensor, target1: torch.Tensor,
                att2_list: Dict[int, torch.Tensor], att2_mask: torch.Tensor, target2: torch.Tensor, record=False):
        """att shape (bs, 4, head, n_obj); target is in ce format"""
        output_dict = {}
        loss = 0
        bs = att1_mask.shape[0]
        if not self.per_head and not self.head_average:
            att1_mask = att1_mask.view(bs, 4, -1)
            att2_mask = att2_mask.view(bs, 4, -1)
        elif not self.per_head and self.head_average:
            att1_mask = att1_mask[:, :, 0]
            att2_mask = att2_mask[:, :, 0]
        else:
            assert False
        for layer in self.layers:
            att1 = att1_list[layer]  # head layer
            att2 = att2_list[layer]
            if not self.per_head and not self.head_average:
                att1 = att1.view(bs, 4, -1)  # head layer
                att2 = att2.view(bs, 4, -1)
            elif not self.per_head and self.head_average:
                att1 = att1.mean(dim=2)
                att2 = att2.mean(dim=2)
            else:
                assert False

            target = 4 * target1 + target2
            sim = self.sim_fn(att1, att1_mask, att2, att2_mask, target).view(bs, 16)
            _loss = self.loss_fn(sim, target)

            loss += _loss
            output_dict[f'loss_{layer}'] = _loss.item()
            output_dict[f'acc_{layer}'] = (sim.argmax(-1) == target).float().mean().item()
            if record:
                self.loss[layer] += output_dict[f'loss_{layer}'] * bs
                self.acc[layer] += output_dict[f'acc_{layer}'] * bs
        if record:
            self.count += bs
        loss /= len(self.layers)
        return loss, output_dict

    def reset(self):
        self.count = 0
        for layer in self.layers:
            self.loss[layer] = 0.
            self.acc[layer] = 0.

    def get_metrics(self, reset: bool = False):
        metrics = {}
        if self.count > 0:
            for layer in self.layers:
                metrics[f'loss_{layer}'] = self.loss[layer]/self.count
                metrics[f'acc_{layer}'] = self.acc[layer]/self.count
        else:
            for layer in self.layers:
                metrics[f'loss_{layer}'] = 0.
                metrics[f'acc_{layer}'] = 0.
        if reset:
            self.reset()
        return metrics


class CosineEmbeddingAlignLoss4(nn.Module):
    def __init__(self, margin=0.2, layers=(11, )):
        super(CosineEmbeddingAlignLoss4, self).__init__()
        self.margin = margin
        self.layers = layers
        self._loss = nn.CosineEmbeddingLoss(margin=margin)
        self.acc1 = dict((layer, 0.) for layer in layers)
        self.acc2 = dict((layer, 0.) for layer in layers)
        self.loss1 = dict((layer, 0.) for layer in layers)
        self.loss2 = dict((layer, 0.) for layer in layers)
        self.count = 0

    def forward(self, att1_list: Dict[int, torch.Tensor], att1_mask: torch.Tensor, target1: torch.Tensor,
                att2_list: Dict[int, torch.Tensor], att2_mask: torch.Tensor, target2: torch.Tensor, record=False):
        """att shape (bs, 4, head, n_obj); target is in ce format"""
        output_dict = {}
        loss = 0
        bs = att1_mask.shape[0]
        att1_mask = att1_mask.view(bs, 4, -1)
        att2_mask = att2_mask.view(bs, 4, -1)
        for layer in self.layers:
            att1 = att1_list[layer]
            att2 = att2_list[layer]
            att1 = att1.view(bs, 4, -1)  # head layer
            att2 = att2.view(bs, 4, -1)
            att1_gt = att1[torch.arange(bs), target1.long(), None].repeat(1, 4, 1)
            att2_gt = att2[torch.arange(bs), target2.long(), None].repeat(1, 4, 1)
            label1 = -att1.new_ones((bs, 4))
            label1[torch.arange(bs), target1.long()] = 1
            label2 = -att2.new_ones((bs, 4))
            label2[torch.arange(bs), target2.long()] = 1

            loss1 = self._loss(att1.view(bs*4, -1), att2_gt.view(bs*4, -1), label1.view(-1))
            loss2 = self._loss(att2.view(bs*4, -1), att1_gt.view(bs*4, -1), label2.view(-1))

            loss += (loss1 + loss2) / 2
            output_dict[f'loss1_{layer}'] = loss1.item()
            output_dict[f'loss2_{layer}'] = loss2.item()

            sim1 = F.cosine_similarity(att1, att2_gt, dim=-1)
            sim2 = F.cosine_similarity(att2, att1_gt, dim=-1)
            output_dict[f'acc1_{layer}'] = (sim1.argmax(-1) == target1).float().mean().item()
            output_dict[f'acc2_{layer}'] = (sim2.argmax(-1) == target2).float().mean().item()
            if record:
                self.loss1[layer] += output_dict[f'loss1_{layer}'] * bs
                self.loss2[layer] += output_dict[f'loss2_{layer}'] * bs
                self.acc1[layer] += output_dict[f'acc1_{layer}'] * bs
                self.acc2[layer] += output_dict[f'acc2_{layer}'] * bs
        if record:
            self.count += bs
        loss /= len(self.layers)
        return loss, output_dict

    def reset(self):
        self.count = 0
        for layer in self.layers:
            self.loss1[layer] = 0.
            self.loss2[layer] = 0.
            self.acc1[layer] = 0.
            self.acc2[layer] = 0.

    def get_metrics(self, reset: bool = False):
        metrics = {}
        if self.count > 0:
            for layer in self.layers:
                metrics[f'loss1_{layer}'] = self.loss1[layer] / self.count
                metrics[f'loss2_{layer}'] = self.loss2[layer] / self.count
                metrics[f'acc1_{layer}'] = self.acc1[layer] / self.count
                metrics[f'acc2_{layer}'] = self.acc2[layer] / self.count
        else:
            for layer in self.layers:
                metrics[f'loss1_{layer}'] = 0.
                metrics[f'loss2_{layer}'] = 0.
                metrics[f'acc1_{layer}'] = 0.
                metrics[f'acc2_{layer}'] = 0.
        if reset:
            self.reset()
        return metrics


SIMILARITY_FUNCTIONS = {
    'inner_product': InnerProduct, 'cosine': CosineSimilarity, 'cosface': AddMarginProduct, 'arcface': ArcMarginProduct,
    'apprank': AppRank, 'listmle': ListMLE, 'spearman': Spearman,
    'mse': NegativeMSE, 'sse': NegativeSSE, 'kl': NegativeKLDiv
}
LOSS_FUNCTIONS = {'ce': nn.CrossEntropyLoss, 'mrl': MarginRankingLoss, 'embed': EmbeddingLoss,
                  'bce': BinaryCrossEntropyLoss, 'focal': FocalLoss}
ALIGN_MODELS = {'align4': AlignLoss, 'align16': AlignLoss16, 'cosembedding4': CosineEmbeddingAlignLoss4}


def get_align_model(model_params, sim_fn_params=None, loss_fn_params=None):
    model_name = model_params.pop('name')
    align_model = ALIGN_MODELS[model_name]
    sim_fn = None
    if sim_fn_params:
        sim_fn = SIMILARITY_FUNCTIONS[sim_fn_params.pop('name')](**sim_fn_params)
    loss_fn = None
    if loss_fn_params:
        loss_fn = LOSS_FUNCTIONS[loss_fn_params.pop('name')](**loss_fn_params)
    if model_name in ('cosembedding4', ):
        model = align_model(**model_params)
    else:
        model = align_model(sim_fn=sim_fn, loss_fn=loss_fn, **model_params)
    return model


