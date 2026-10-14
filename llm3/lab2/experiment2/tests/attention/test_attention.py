

"""将手写注意力逐项对拍 PyTorch 标准实现。"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from lab.implementation import get_attention_module

attention = get_attention_module()
MultiHeadAttention = attention.MultiHeadAttention
causal_scaled_dot_product_attention = attention.causal_scaled_dot_product_attention
make_causal_mask = attention.make_causal_mask
scaled_dot_product_attention = attention.scaled_dot_product_attention


DTYPE = torch.float64


def _copy_reference_parameters(
    handwritten: MultiHeadAttention, reference: nn.MultiheadAttention
) -> None:
    """把手写模型的四组参数复制到 PyTorch 标准模块。"""

    with torch.no_grad():
        reference.in_proj_weight.copy_(
            torch.cat(
                [
                    handwritten.q_proj.weight,
                    handwritten.k_proj.weight,
                    handwritten.v_proj.weight,
                ]
            )
        )
        if reference.in_proj_bias is not None:
            reference.in_proj_bias.copy_(
                torch.cat(
                    [
                        handwritten.q_proj.bias,
                        handwritten.k_proj.bias,
                        handwritten.v_proj.bias,
                    ]
                )
            )
        reference.out_proj.weight.copy_(handwritten.out_proj.weight)
        if reference.out_proj.bias is not None:
            reference.out_proj.bias.copy_(handwritten.out_proj.bias)


def test_scaled_dot_product_attention_matches_pytorch() -> None:
    #torch.manual_seed(0)
    q = torch.randn(2, 3, 4, 5, dtype=DTYPE)
    k = torch.randn(2, 3, 6, 5, dtype=DTYPE)
    v = torch.randn(2, 3, 6, 7, dtype=DTYPE)

    actual = scaled_dot_product_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_boolean_and_additive_masks_match_pytorch() -> None:
   # torch.manual_seed(1)
    q = torch.randn(2, 2, 4, 8, dtype=DTYPE)
    k = torch.randn(2, 2, 4, 8, dtype=DTYPE)
    v = torch.randn(2, 2, 4, 6, dtype=DTYPE)
    boolean_mask = make_causal_mask(4)
    additive_mask = torch.zeros(4, 4, dtype=DTYPE).masked_fill(
        ~boolean_mask, float("-inf")
    )

    for mask in (boolean_mask, additive_mask):
        actual = scaled_dot_product_attention(q, k, v, mask)
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_causal_attention_matches_pytorch_and_cannot_see_future() -> None:
    #torch.manual_seed(2)
    q = torch.randn(2, 3, 5, 4, dtype=DTYPE)
    k = torch.randn(2, 3, 5, 4, dtype=DTYPE)
    v = torch.randn(2, 3, 5, 4, dtype=DTYPE)

    actual, weights = causal_scaled_dot_product_attention(
        q, k, v, return_attention=True
    )
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(weights.triu(diagonal=1)) == 0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))


def test_scaled_attention_gradients_match_pytorch() -> None:
    #torch.manual_seed(3)
    inputs = [
        torch.randn(2, 2, 4, 6, dtype=DTYPE, requires_grad=True),
        torch.randn(2, 2, 5, 6, dtype=DTYPE, requires_grad=True),
        torch.randn(2, 2, 5, 7, dtype=DTYPE, requires_grad=True),
    ]
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]

    scaled_dot_product_attention(*inputs).square().sum().backward()
    F.scaled_dot_product_attention(*reference_inputs).square().sum().backward()

    for actual, expected in zip(inputs, reference_inputs):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("is_causal", [False, True])
def test_multihead_self_attention_matches_pytorch(is_causal: bool) -> None:
    #torch.manual_seed(4)
    handwritten = MultiHeadAttention(12, 3, dropout=0.0).to(DTYPE)
    reference = nn.MultiheadAttention(
        12, 3, dropout=0.0, batch_first=True
    ).to(DTYPE)
    _copy_reference_parameters(handwritten, reference)
    x = torch.randn(2, 5, 12, dtype=DTYPE)

    actual_output, actual_weights = handwritten(
        x,
        is_causal=is_causal,
        need_weights=True,
        average_attn_weights=False,
    )
    # nn.MultiheadAttention 的布尔 mask 语义相反：True 表示禁止关注。
    reference_mask = ~make_causal_mask(5) if is_causal else None
    expected_output, expected_weights = reference(
        x,
        x,
        x,
        attn_mask=reference_mask,
        need_weights=True,
        average_attn_weights=False,
    )

    torch.testing.assert_close(actual_output, expected_output, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-11, atol=1e-11)


def test_multihead_cross_attention_and_padding_match_pytorch() -> None:
    #torch.manual_seed(5)
    handwritten = MultiHeadAttention(8, 2, dropout=0.0).to(DTYPE)
    reference = nn.MultiheadAttention(8, 2, batch_first=True).to(DTYPE)
    _copy_reference_parameters(handwritten, reference)
    query = torch.randn(2, 3, 8, dtype=DTYPE)
    memory = torch.randn(2, 5, 8, dtype=DTYPE)
    padding = torch.tensor(
        [[False, False, False, True, True], [False, False, False, False, True]]
    )

    actual, actual_weights = handwritten(
        query,
        memory,
        memory,
        key_padding_mask=padding,
        need_weights=True,
    )
    expected, expected_weights = reference(
        query,
        memory,
        memory,
        key_padding_mask=padding,
        need_weights=True,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-11, atol=1e-11)
    assert torch.count_nonzero(actual_weights.masked_select(padding[:, None, :])) == 0


def test_multihead_parameters_receive_gradients() -> None:
    #torch.manual_seed(6)
    model = MultiHeadAttention(8, 2)
    x = torch.randn(2, 4, 8, requires_grad=True)

    model(x, is_causal=True).sum().backward()

    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_invalid_head_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="整除"):
        MultiHeadAttention(embed_dim=10, num_heads=3)
