"""
Usage:
    cd attentions_pytorch; pytest test/test_attentions.py
"""

import pytest
import torch

from src.attentions import DotProductAttention

def dot_product_attention_test():
    print(f"Start Dot-Product Attention test")

    attn = DotProductAttention()

    # Make dummy data.
    query_length = 2
    value_length = 3
    embed_dim = 4
    query = torch.randn(query_length, embed_dim)
    value = torch.randn(value_length, embed_dim)

    attn_value, attn_weights = attn(query, value)
    print(f"Attention Value: {attn_value}")
    print(f"Attention Weights: {attn_weights}")

    assert attn_value.shape == (query_length, embed_dim)
    assert attn_weights.shape == (query_length, value_length)
    
    for i in range(query_length):
        assert round(float(attn_weights[i].sum()), 0) == 1.0


def test_main():
    dot_product_attention_test()

