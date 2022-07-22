import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from typing import Tuple

class DotProductAttention(nn.Module):
    """
    This class is implementation of Dot-Product Attention mechanism
    proposed in "Effective Approaches to Attention-based Neural Machine Translation"(Luong et al., 2015).
    
    1. Compute attention score by dot-products of the query and values.
    2. Compute attention weights by softmax function on attention score.
    3. Compute attenion value(context vector) by weighted sum of attention weights and values.
    4. Return attention value and attention weights.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        query: Tensor,
        value: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query:
                Query Embedding tensors.
            value:
                Value Embedding tensors.

        Returns:
            A tuple of attention value and attention weights about query and value.

        Shape:
            query: :math:`(L_q, E)` where L_q is query length(target sequence length),
                E is embedding dimension.
            value: :math:`(L_v, E)` where L_v is value length(source sequence length),
                E is embedding dimension.
            
            Returns:
                attention value: :math:`(L_q, E)`
                attention weights: :math: `(L_q, L_v)`
        """
        attn_score = torch.mm(query, value.transpose(0, 1))
        attn_weights = F.softmax(attn_score, dim=-1)
        attn_value = torch.mm(attn_weights, value)
        return attn_value, attn_weights


