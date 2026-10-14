"""可直接运行的小型演示：python -m attention_experiment.demo"""

from __future__ import annotations

import sys

import torch

from lab.implementation import get_attention_module

attention = get_attention_module()
MultiHeadAttention = attention.MultiHeadAttention
causal_scaled_dot_product_attention = attention.causal_scaled_dot_product_attention


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    #torch.manual_seed(7)
    batch_size, sequence_length, embed_dim, num_heads = 2, 4, 8, 2
    x = torch.randn(batch_size, sequence_length, embed_dim)

    print("1) 因果缩放点积注意力")
    qkv = x.reshape(batch_size, sequence_length, num_heads, embed_dim // num_heads)
    qkv = qkv.transpose(1, 2)
    output, weights = causal_scaled_dot_product_attention(
        qkv, qkv, qkv, return_attention=True
    )
    print("输出形状:", tuple(output.shape))
    print("第一个样本、第一头的注意力权重:\n", weights[0, 0])

    print("\n2) 多头因果自注意力")
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    mha_output, mha_weights = mha(
        x, is_causal=True, need_weights=True, average_attn_weights=False
    )
    print("输出形状:", tuple(mha_output.shape))
    print("权重形状:", tuple(mha_weights.shape))
    print("未来位置的最大权重:", mha_weights.triu(diagonal=1).max().item())


if __name__ == "__main__":
    main()
