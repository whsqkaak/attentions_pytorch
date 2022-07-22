"""
Usage:
    cd attentions_pytorch; pytest test/test_attentions.py
"""

import pytest
import torch

from src.attentions import (
    DotProductAttention,
    GeneralAttention,
    ScaledDotProductAttention
    )

def attention_test():
    # Prepare dummpy data.
    print(f"Prepare dummy data.")
    
    # Make dummy data.
    N_t = 2
    N_s = 3
    E = 4
    query = torch.randn(N_t, E)
    key = torch.randn(N_s, E)
    value = torch.randn(N_s, E)

    print(f"Start Dot-Product Attention test.")
    attentions = ["dot", "general", "scaled"]

    for type_attention in attentions:
        if type_attention == "dot":
            attn = DotProductAttention()
            attn_value, attn_weights = attn(query, value)
        
        elif type_attention == "general":
            attn = GeneralAttention(E)
            attn_value, attn_weights = attn(query, value)

        elif type_attention == "scaled":
            attn = ScaledDotProductAttention()
            attn_value, attn_weights = attn(query, key, value, attn_mask=None)

        print(f"Attention Value: {attn_value}")
        print(f"Attention Weights: {attn_weights}")

        assert attn_value.shape == (N_t, E)
        assert attn_weights.shape == (N_t, N_s)
    
        for i in range(N_t):
            assert round(float(attn_weights[i].sum()), 0) == 1.0
    

def test_main():
    attention_test()
