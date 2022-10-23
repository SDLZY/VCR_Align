import torch


def apprank(input1: torch.Tensor, mask1: torch.Tensor,
            input2: torch.Tensor, mask2: torch.Tensor, alpha: float = 1.0, label=None):
    """input: (bs, heads, nobj), mask: (bs, heads, nobj), input2 is target"""
    # import pdb; pdb.set_trace()
    bs, num_heads, _ = input1.shape
    # mask1 = mask1.unsqueeze(1)
    # mask2 = mask2.unsqueeze(1)
    input1 = input1 * mask1.float()
    input2 = (input2 * mask2.float()).detach()

    diff = input1.unsqueeze(-1) - input1.unsqueeze(-2)
    diff = torch.sigmoid(-diff * alpha)  # b, heads, n, n

    mask1_ = mask1.unsqueeze(-1) * mask1.unsqueeze(-2)  # b, 1, n, n
    pi = 0.5 + (diff * mask1_.float()).sum(-1)  # b, heads, n
    app_dcg = input2.exp() / (1 + pi).log()
    app_dcg = (app_dcg * mask1.float()).sum(-1)

    input2 = torch.where(mask2 > 0, input2, input2.new_full((1,), -1E5))
    input2_sort, input2_sorted_indices = input2.sort(-1, descending=True)
    mask2 = torch.gather(mask2, dim=-1, index=input2_sorted_indices)
    idcg = input2_sort.exp() / torch.arange(2, input2.shape[-1] + 2, device=input2.device).float().log()
    idcg = (idcg * mask2.float()).sum(-1)
    ndcg = app_dcg / idcg
    # ndcg = ndcg.view(bs, num_heads, -1)
    return 1 - ndcg