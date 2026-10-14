"""缩放点积注意力、因果掩码和多头注意力的手写实现。

核心计算只依赖 PyTorch 的基础张量操作，没有调用
``torch.nn.functional.scaled_dot_product_attention`` 或
``torch.nn.MultiheadAttention``。后两者只在测试中作为标准答案使用。

本文件采用和 ``torch.nn.functional.scaled_dot_product_attention`` 一致的
布尔掩码语义：``True`` 表示该位置允许被关注，``False`` 表示屏蔽。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def make_causal_mask(
    query_length: int,
    key_length: int | None = None,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """返回形状为 ``(query_length, key_length)`` 的下三角布尔掩码。

    第 ``i`` 个 query 只能看到位置不大于 ``i`` 的 key。非方阵时也采用
    PyTorch SDPA 的左上对齐（upper-left）定义。
    """

    if query_length < 0:
        raise ValueError("query_length 不能为负数")
    if key_length is None:
        key_length = query_length
    if key_length < 0:
        raise ValueError("key_length 不能为负数")

    # TODO[ATTN-02]: 返回 (Lq, Lk) bool 张量，j <= i 的位置为 True。
    # 支持 device 与非方阵左上对齐。
    mask = torch.tril(torch.ones(query_length,key_length,dtype=torch.bool,device=device))
    return mask
    raise NotImplementedError("ATTN-02: 请实现因果掩码")


def _validate_qkv(query: Tensor, key: Tensor, value: Tensor) -> None:
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query、key、value 至少需要二维：(..., sequence, feature)")
    if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
        raise ValueError("query、key、value 的批次/头维度必须一致")
    if query.size(-1) != key.size(-1):
        raise ValueError("query 和 key 的特征维度必须一致")
    if key.size(-2) != value.size(-2):
        raise ValueError("key 和 value 的序列长度必须一致")
    if query.size(-1) == 0:
        raise ValueError("query/key 的特征维度必须大于 0")
    if not (query.is_floating_point() and key.is_floating_point() and value.is_floating_point()):
        raise TypeError("query、key、value 必须是浮点张量")


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None = None,
    *,
    dropout_p: float = 0.0,
    training: bool = True,
    scale: float | None = None,
    return_attention: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """用基础张量操作实现缩放点积注意力。

    参数形状为 ``(..., Lq, Dk)``、``(..., Lk, Dk)``、
    ``(..., Lk, Dv)``。``attention_mask`` 必须能广播到分数矩阵
    ``(..., Lq, Lk)``：布尔掩码中 ``True`` 表示允许关注，浮点掩码则
    直接加到注意力分数上（通常用 ``-inf`` 屏蔽位置）。
    """

    _validate_qkv(query, key, value)
    if not 0.0 <= dropout_p <= 1.0:
        raise ValueError("dropout_p 必须位于 [0, 1]")

    # TODO[ATTN-01]: QK^T / sqrt(Dk) -> 掩码 -> 最后一维 softmax -> AV。
    # 支持自定义 scale、bool/浮点掩码、dropout 和 return_attention。
    # 整行屏蔽时权重为 0，避免 NaN；不得调用 PyTorch 标准注意力。

    # 计算出d_k
    d_k = key.size(-1)

    # 算QK^T，不能直接转置（多头注意力时K是四维张量，最后两个维度才是原先的K的维度）
    scores = query@key.transpose(-2,-1)

    if scale is not None:
        scores*=scale
    else:
        scores/=math.sqrt(d_k)
    if attention_mask is not None:
        if attention_mask.dtype ==torch.bool:
            scores = torch.masked_fill(scores,~attention_mask,float("-inf"))
        else:
            scores = scores + attention_mask

    row_visible = torch.isfinite(scores).any(dim=-1,keepdim=True)
    scores = torch.masked_fill(scores,~row_visible,0.0)

    weights = torch.softmax(scores,dim=-1)
    weights = torch.masked_fill(weights,~row_visible,0.0)

    weights = torch.nn.functional.dropout(weights,dropout_p,training=training)

    output = weights@value
    if return_attention:
        return output,weights
    return output
    # raise NotImplementedError("ATTN-01: 请实现缩放点积注意力")


def causal_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    dropout_p: float = 0.0,
    training: bool = True,
    scale: float | None = None,
    return_attention: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """应用显式因果掩码的缩放点积注意力。"""

    # TODO[ATTN-03]: 用 make_causal_mask 构建掩码，再复用手写 SDPA。
    # 将 dropout_p、training、scale、return_attention 传递给底层。
    causal_mask = make_causal_mask(query.size(-2),key.size(-2),device=query.device)
    return scaled_dot_product_attention(query,key,value,causal_mask,dropout_p=dropout_p,training=training,scale=scale,return_attention=return_attention)

    raise NotImplementedError("ATTN-03: 请组合因果注意力")


class MultiHeadAttention(nn.Module):
    """从零组装的 batch-first 多头注意力。

    输入和输出均使用 ``(batch, sequence, embed_dim)``。布尔
    ``attention_mask`` 中 ``True`` 表示允许关注；布尔
    ``key_padding_mask`` 则遵循 PyTorch MHA 的常见约定，``True`` 表示
    padding，需要被屏蔽。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim 必须大于 0")
        if num_heads <= 0:
            raise ValueError("num_heads 必须大于 0")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout 必须位于 [0, 1]")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # TODO[ATTN-04]: 建立 q_proj/k_proj/v_proj/out_proj 四个 nn.Linear。
        # 输入输出维度均为 embed_dim，遵循 bias 参数，调用 _reset_parameters。
        self.q_proj = nn.Linear(embed_dim,embed_dim,bias)
        self.k_proj = nn.Linear(embed_dim,embed_dim,bias)
        self.v_proj = nn.Linear(embed_dim,embed_dim,bias)
        self.out_proj = nn.Linear(embed_dim,embed_dim,bias)
        self._reset_parameters()
        # raise NotImplementedError("ATTN-04: 请定义 Q/K/V 和输出投影")

    def _reset_parameters(self) -> None:
        # 初始化辅助代码已提供；对拍测试会显式复制参数，不依赖随机初始化一致。
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        # TODO[ATTN-05]: (B, L, E) -> (B, H, L, D)，D = E // H。
        return tensor.view(tensor.size(0),tensor.size(1),
                           tensor.size(2)//self.head_dim,self.head_dim).transpose(1,2)

        # raise NotImplementedError("ATTN-05: 请拆分多头")

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        # TODO[ATTN-06]: (B, H, L, D) -> (B, L, E)，注意维度顺序/连续性。
        return tensor.transpose(1,2).flatten(2,3)
        # raise NotImplementedError("ATTN-06: 请拼接多头")

    @staticmethod
    def _expand_attention_mask(
        mask: Tensor,
        batch_size: int,
        num_heads: int,
        query_length: int,
        key_length: int,
    ) -> Tensor:
        if mask.ndim == 2:
            if mask.shape != (query_length, key_length):
                raise ValueError("二维 attention_mask 的形状必须是 (Lq, Lk)")
            return mask.unsqueeze(0).unsqueeze(0)
        if mask.ndim == 3:
            if mask.shape != (batch_size, query_length, key_length):
                raise ValueError("三维 attention_mask 的形状必须是 (B, Lq, Lk)")
            return mask.unsqueeze(1)
        if mask.ndim == 4:
            expected = (batch_size, num_heads, query_length, key_length)
            if mask.shape != expected:
                raise ValueError("四维 attention_mask 的形状必须是 (B, H, Lq, Lk)")
            return mask
        raise ValueError("attention_mask 只能是二维、三维或四维")

    def _build_mask(
        self,
        *,
        attention_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool,
        batch_size: int,
        query_length: int,
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        combined: Tensor | None = None

        if attention_mask is not None:
            if attention_mask.device != device:
                raise ValueError("attention_mask 必须和输入位于同一设备")
            if attention_mask.dtype != torch.bool and not attention_mask.is_floating_point():
                raise TypeError("attention_mask 只能是布尔或浮点张量")
            combined = self._expand_attention_mask(
                attention_mask,
                batch_size,
                self.num_heads,
                query_length,
                key_length,
            )

        needs_boolean_mask = is_causal or key_padding_mask is not None
        if needs_boolean_mask:
            allowed = torch.ones(
                (batch_size, 1, query_length, key_length),
                dtype=torch.bool,
                device=device,
            )
            if is_causal:
                allowed &= make_causal_mask(query_length, key_length, device=device)
            if key_padding_mask is not None:
                if key_padding_mask.device != device:
                    raise ValueError("key_padding_mask 必须和输入位于同一设备")
                if key_padding_mask.dtype != torch.bool:
                    raise TypeError("key_padding_mask 必须是布尔张量")
                if key_padding_mask.shape != (batch_size, key_length):
                    raise ValueError("key_padding_mask 的形状必须是 (B, Lk)")
                allowed &= ~key_padding_mask[:, None, None, :]

            if combined is None:
                combined = allowed
            elif combined.dtype == torch.bool:
                combined = combined & allowed
            else:
                combined = combined.to(dtype=dtype).masked_fill(~allowed, float("-inf"))

        return combined

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        *,
        attention_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """执行 self-attention 或 cross-attention。"""

        key = query if key is None else key
        value = key if value is None else value
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query、key、value 必须是 (B, L, E) 三维张量")
        if query.size(0) != key.size(0) or key.size(0) != value.size(0):
            raise ValueError("query、key、value 的 batch size 必须一致")
        if key.size(1) != value.size(1):
            raise ValueError("key 和 value 的序列长度必须一致")
        if query.size(-1) != self.embed_dim or key.size(-1) != self.embed_dim:
            raise ValueError("query 和 key 的最后一维必须等于 embed_dim")
        if value.size(-1) != self.embed_dim:
            raise ValueError("value 的最后一维必须等于 embed_dim")

        batch_size, query_length, _ = query.shape
        key_length = key.size(1)
        # TODO[ATTN-07]: 投影 -> 拆头 -> _build_mask -> 手写 SDPA -> 拼头 -> 输出投影。
        # need_weights=False 仅返回输出；否则按 average_attn_weights 决定是否平均头。
        # dropout 在 eval 模式应关闭；权重形状为 (B,H,Lq,Lk) 或 (B,Lq,Lk)。
        device = query.device
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        q_splited = self._split_heads(query)
        k_splited = self._split_heads(key)
        v_splited = self._split_heads(value)

        mask = self._build_mask(attention_mask=attention_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                batch_size=batch_size,
                                query_length=query_length,
                                key_length=key_length,
                                device=device,
                                dtype=query.dtype)

        if need_weights:
            output,weights = scaled_dot_product_attention(q_splited,
                                                          k_splited,
                                                          v_splited,
                                                          mask,
                                                          dropout_p=self.dropout,
                                                          training=self.training,
                                                          scale=None,
                                                          return_attention=need_weights)
        else:
            output = scaled_dot_product_attention(q_splited,
                                                  k_splited,
                                                  v_splited,
                                                  mask,
                                                  dropout_p=self.dropout,
                                                  training=self.training,
                                                  scale=None,
                                                  return_attention=need_weights)
            weights = None
        output_merged = self._merge_heads(output)
        output_merged = self.out_proj(output_merged)

        if weights is not None and average_attn_weights:
            weights = weights.mean(dim = 1)

        if need_weights:
            return output_merged,weights
        else:
            return output_merged
            
        # raise NotImplementedError("ATTN-07: 请完成多头注意力前向传播")
