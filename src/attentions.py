import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from typing import Tuple, Optional

class DotProductAttention(nn.Module):
    """
    This class is implementation of Dot-Product Attention mechanism
    proposed in "Effective Approaches to Attention-based Neural Machine Translation"(Luong et al., 2015).
    
    1. Compute attention score by dot-products of the query and values.
    2. Compute attention weights by softmax function on attention score.
    3. Compute attenion value(context vector) by weighted sum of attention weights and values.
    4. Return attention value and attention weights.
    """
    # TODO: To update batch version.

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


class ScaledDotProductAttention(nn.Module):
    """
    This class is implementation of Scaled Dot-Product Attention mechanism
    proposed in "Attention is All you Need"(Vaswani et al., 2017).

    Computes scaled dot product attntion on query, key and value tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query:
                Query Embedding tensors.
            key:
                Key Embedding tensors.
            value:
                Value Embedding tensors.
            attn_mask:
                Optional tensor containing mask values to be added to calculated
                attention.

        Returns:
            A tuple of attention value and attention weights.

        Shape:
            query: :math:`(N_t, E)` where N_t is the target sequence length,
                and E is embedding dimension.
            key: :math:`(N_s, E)` where N_s is the source sequence length,
                and E is embedding dimension.
            value: :math:`(N_s, E)` where N_s is the source sequence length,
                and E is embedding dimension.
            attn_mask: :math:`(N_t, N_s)` where N_t is the target sequence length,
                and N_s is the source sequence length.

            Returns:
                attention value: :math:`(N_t, E)`
                attention weights: :math:`(N_t, N_s)`
        """
        embed_dim = query.shape[1]

        attn_score = torch.mm(query, key.transpose(0, 1))
        attn_score = attn_score / math.sqrt(embed_dim)
        
        if attn_mask is not None:
            attn_score.masked_fill(mask, -float('Inf'))

        attn_weights = F.softmax(attn_score, dim=-1)
        attn_value = torch.mm(attn_weights, value)
        
        return attn_value, attn_weights



