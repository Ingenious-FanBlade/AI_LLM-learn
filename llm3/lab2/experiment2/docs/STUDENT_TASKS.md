# 学生实验任务与补全清单

## 一、环境与工作范围

使用 Python 3.11，安装 `requirements.txt` 中依赖。在项目根目录运行命令。
只补全 `lab/student/bpe.py`、`lab/student/attention.py` 的编号 TODO。
辅助代码已提供；保持类名、函数签名、参数名、返回值不变。
删除相应 `raise NotImplementedError(...)` 并填写实现，不能只删异常或改成 pass。

## 二、BPE：按顺序补全 5 项

| 编号 | 位置 | 需实现的核心能力 |
| --- | --- | --- |
| BPE-01 | _count_pairs | 相邻 pair 统计，允许统计重叠出现 |
| BPE-02 | _merge_pair | 从左到右、不重叠地合并，不修改输入 |
| BPE-03 | train | 统计→最高频 pair→创建 token→更新序列的训练循环 |
| BPE-04 | encode_bytes | 按训练 merge rank 编码，不在输入上重新训练 |
| BPE-05 | decode_bytes | token 字节拼接、特殊 token 处理与非法 id 检查 |

基础词表为 256 个字节；最终目标词表包含 4 个特殊 token。
目标 512/1024/2048 分别最多进行 252/764/1788 次合并。
短语料没有可合并 pair 时提前结束，记录实际词表大小。
频率并列按 pair 的 id 字典序选择较小者；新 token id 从 256 连续增加。

## 三、注意力：按顺序补全 7 项

| 编号 | 位置 | 需实现的核心能力 |
| --- | --- | --- |
| ATTN-01 | scaled_dot_product_attention | QKᵀ / √Dk、掩码、Softmax、AV |
| ATTN-02 | make_causal_mask | j ≤ i 的布尔掩码，包括非方阵 |
| ATTN-03 | causal_scaled_dot_product_attention | 复用掩码与手写点积注意力 |
| ATTN-04 | MultiHeadAttention.__init__ | Q/K/V 与输出的四个线性投影 |
| ATTN-05 | _split_heads | (B,L,E) → (B,H,L,E/H) |
| ATTN-06 | _merge_heads | (B,H,L,E/H) → (B,L,E) |
| ATTN-07 | forward | 投影、拆头、计算注意力、拼头、输出投影及权重返回 |

只能使用基础张量操作、`nn.Linear` 等组装算法。
**禁止在核心实现中直接调用 PyTorch SDPA 或 MultiheadAttention，也不能导入参考答案。**
标准实现只在测试中充当对拍对象。

掩码约定：`attention_mask=True` 表示允许关注；`key_padding_mask=True` 表示屏蔽 padding。
Softmax 在 key 所在的最后一维执行。无 dropout 时，有可见 key 的行权重和为 1；
整行屏蔽时输出与权重为 0。未来位置权重为 0。
多头必须满足 `embed_dim % num_heads == 0`。
对拍时使用相同输入、相同权重、关闭 dropout，而非只设置相同随机种子。

## 四、运行与自检

```powershell
python -m pytest tests/bpe -v
python -m pytest tests/attention -v
python -m pytest tests/bpe tests/attention --implementation student -v
python main.py attention-demo
python main.py bpe-cli --help
```

测试默认选择学生版。未补全时失败是正常的。
建议按 TODO 逐项实现并使用 `-k 测试名` 定位错误。

## 五、最终实验要求与提交

完成全部 12 项 TODO；不能修改验收测试来使结果通过，也不能用参考实现绕过作业。

BPE 分别设置词表目标 512、1024、2048，使用相同训练语料和独立的相同测试文本，
统计原始 UTF-8 字节数、实际词表大小、token 数与
CR = 原始 UTF-8 字节数 / token 数（非空文本，不加入 BOS/EOS）。
比较 tiktoken 的 token 数、压缩率和分词边界，分析训练语料、词表、规则差异；
回答“词表是否越大越好”，不只给出压缩率结论。

提交两个学生实现文件、实验报告及全部核心测试通过的截图。

