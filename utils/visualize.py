import seaborn as sns
import matplotlib.pyplot as plt


def get_attention_drawer(path, size=3):
    def draw_attention(attention_list, attention_mask, num_txt_tokens):
        """
        @param attention_list: List[Dict[probs: torch.Tensor, scores: torch.Tensor]]
        @param attention_mask: torch.Tensor (bs, num_objs)
        @param num_txt_tokens: torch.Tensor (bs), 每个example中文本token的个数
        """
        num_layers = len(attention_list)
        num_heads = attention_list[0]['probs'].shape[1]
        fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads*size, num_layers*size), sharex=True, sharey=True)
        for layer, attention in enumerate(attention_list):
            attention = attention['probs']
            if len(attention.shape) == 4:
                attention = attention[0]  # num_head, n, n
                if len(attention_mask.shape) == 2:
                    attention_mask = attention_mask[0]
                if len(num_txt_tokens.shape) == 1:
                    num_txt_tokens = num_txt_tokens[0]
            valid = attention_mask.nonzero().squeeze()
            attention = attention[:, valid][:, :, valid]
            for head in range(attention.shape[0]):
                sns.heatmap(attention[head].data.cpu().numpy(), square=True, vmin=0, vmax=1, cbar=False, ax=axes[layer, head])
                axes[layer, head].axhline(num_txt_tokens.item(), color='green', alpha=0.5)
                axes[layer, head].axvline(num_txt_tokens.item(), color='green', alpha=0.5)
        plt.savefig(path)
    return draw_attention

