"""一个从零实现、便于教学的字节级 BPE 分词器。

这个文件只使用 Python 标准库。核心目标不是追求工业级速度，而是把 BPE 的
三个关键过程清楚地展示出来：

1. train：把 UTF-8 字节作为初始 token，反复合并训练语料中最高频的相邻 pair；
2. encode：严格按训练阶段学到的 merge 顺序编码新文本；
3. decode：把 token 对应的 bytes 拼接起来，再按 UTF-8 解码回字符串。

初始 token id 0~255 与单个字节值一一对应，因此任何合法 UTF-8 文本都能编码。
项目仍保留 PAD/BOS/EOS/UNK 四个特殊 token，便于后续神经网络批处理和序列建模；
其中 UNK 在正常字节级编码中不会被自动使用。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


class ByteLevelBPE:
    """手写字节级 Byte Pair Encoding 分词器。

    Attributes:
        vocab:
            token id -> token 原始字节串。例如 97 -> b"a"，256 -> b"th"。
        merges:
            (左 token id, 右 token id) -> 合并后 token id。

    新 token id 从 256 开始依次增加。由于 id 的增加顺序就是训练时学到合并规则
    的顺序，所以编码时选择 id 最小的可用 merge，等价于选择 rank 最小的规则。
    """

    BYTE_VOCAB_SIZE = 256
    MODEL_VERSION = 2
    SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>")

    def __init__(self) -> None:
        """创建带有 256 个基础字节 token 的分词器。"""
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.special_tokens: dict[str, int] = {}
        self.id_to_special: dict[int, str] = {}
        self._reset()

    def _reset(self) -> None:
        """清空训练结果，并重新建立 0~255 的基础字节词表。"""
        self.vocab = {byte_value: bytes([byte_value]) for byte_value in range(256)}
        self.merges = {}
        self.special_tokens = {}
        self.id_to_special = {}

    def _assign_special_tokens(self) -> None:
        """把特殊 token 连续放在所有普通 BPE token 之后。"""
        first_special_id = len(self.vocab)
        self.special_tokens = {
            name: first_special_id + offset
            for offset, name in enumerate(self.SPECIAL_TOKENS)
        }
        self.id_to_special = {
            token_id: name for name, token_id in self.special_tokens.items()
        }

    @staticmethod
    def _render_progress(
        completed: int,
        total: int,
        pair_count: int,
        sequence_length: int,
        started_at: float,
    ) -> None:
        """在同一终端行绘制训练进度、速度和预计剩余时间。"""
        ratio = completed / total if total else 1.0
        bar_width = 30
        filled = min(bar_width, int(ratio * bar_width))
        bar = "█" * filled + "░" * (bar_width - filled)
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        merges_per_second = completed / elapsed if completed else 0.0
        eta = (
            (total - completed) / merges_per_second
            if merges_per_second > 0
            else 0.0
        )
        print(
            f"\r训练进度 [{bar}] {ratio:6.1%} "
            f"merge={completed:>3}/{total:<3} "
            f"pair频次={pair_count:<6} "
            f"当前序列={sequence_length:,} "
            f"ETA={eta:5.1f}s",
            end="",
            flush=True,
        )

    @staticmethod
    def _count_pairs(token_ids: Sequence[int]) -> Counter[tuple[int, int]]:
        """统计序列里所有相邻 token pair 的出现次数。

        例如 [1, 2, 1, 2] 中：
        (1, 2) 出现 2 次，(2, 1) 出现 1 次。
        """
        # TODO[BPE-01]: 统计所有相邻 pair，包括重叠出现；返回 Counter。
        c = Counter()
        for i in range(len(token_ids)-1):
            c[(token_ids[i],token_ids[i+1])]+=1
        return c;

        raise NotImplementedError("BPE-01: 请实现相邻 pair 统计")

    @staticmethod
    def _merge_pair(
        token_ids: Sequence[int], pair: tuple[int, int], new_id: int
    ) -> list[int]:
        """从左到右把指定 pair 的所有不重叠出现合并成 new_id。

        对 [1, 1, 1] 合并 (1, 1) 时，结果是 [new_id, 1]，而不是尝试让中间
        的 1 同时参与两次合并。这正是 BPE 合并所需的“不重叠”语义。
        """
        # TODO[BPE-02]: 从左到右扫描，匹配成功前进 2，否则前进 1。
        # 不修改输入序列；注意空序列和重叠 pair。
        n = len(token_ids)
        new_id_list = list()
        i = 0
        while i < n:
            if i<n-1 and token_ids[i]==pair[0] and token_ids[i+1]==pair[1]:
                new_id_list.append(new_id)
                i+=2
            else :
                new_id_list.append(token_ids[i])
                i+=1
        return new_id_list
        raise NotImplementedError("BPE-02: 请实现不重叠合并")

    def train(
        self,
        text: str,
        vocab_size: int,
        show_progress: bool = False,
    ) -> None:
        """在一段文本上训练 BPE 合并规则。

        Args:
            text: UTF-8 训练文本。
            vocab_size: 最终总词表大小，包含基础字节、merge token 和特殊 token。
            show_progress: 为 True 时显示实时进度条、pair 频率、序列长度和 ETA。

        训练步骤：
            1. 用 UTF-8 把字符串变成 bytes；
            2. 每个 byte 的数值直接作为初始 token id；
            3. 统计当前 token 序列中的相邻 pair；
            4. 选择频率最高的 pair，创建一个新 token；
            5. 在整个序列中合并它；
            6. 重复，直到达到目标词表大小，或已经没有相邻 pair。

        频率相同时按 pair 的 token id 字典序选择较小者。这不是 BPE 唯一可用的
        tie-break 规则，但显式规定后可以保证不同运行之间结果完全可复现。
        """
        if not isinstance(text, str):
            raise TypeError("text 必须是 str")
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("vocab_size 必须是整数")
        minimum_vocab_size = self.BYTE_VOCAB_SIZE + len(self.SPECIAL_TOKENS)
        if vocab_size < minimum_vocab_size:
            raise ValueError(
                f"当前配置含 {len(self.SPECIAL_TOKENS)} 个特殊 token，"
                f"vocab_size 不能小于 {minimum_vocab_size}"
            )

        # 每次 train 都从基础字节词表重新开始，避免旧模型规则混入新模型。
        self._reset()

        # bytes 是可迭代的整数序列，范围恰好是 0~255，可直接作为初始 token id。
        token_ids = list(text.encode("utf-8"))
        requested_merges = (
            vocab_size - self.BYTE_VOCAB_SIZE - len(self.SPECIAL_TOKENS)
        )
        started_at = time.perf_counter()
        progress_interval = max(1, requested_merges // 100)

        if show_progress:
            self._render_progress(0, requested_merges, 0, len(token_ids), started_at)

        # TODO[BPE-03]: 实现 requested_merges 轮训练，pair 为空时提前结束。
        # 按频率降序、pair 字典序升序破平局；更新 vocab、merges、token_ids。
        # new_id 从 256 连续增加；进度显示可调用已提供的 _render_progress。
        for merge_index in range(requested_merges):
            pairs = self._count_pairs(token_ids)
            if not pairs:
                break

            best = min(pairs, key = lambda p:(-pairs[p],p))
            best_count = pairs[best]

            new_id = self.BYTE_VOCAB_SIZE+merge_index

            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            token_ids = self._merge_pair(token_ids,best,new_id)

            completed = merge_index + 1
            if show_progress and (completed % progress_interval == 0 or completed == requested_merges):
                self._render_progress(completed,requested_merges,best_count,len(token_ids),started_at)

        if show_progress:
            print()

        # 特殊 token 不参与 pair 统计，训练结束后才追加，因此绝不会被普通 merge 吞掉。
        self._assign_special_tokens()

    def encode_bytes(self, raw: bytes) -> list[int]:
        """编码原始 bytes；encode(text) 是它的 UTF-8 字符串包装。

        注意：编码阶段绝不能重新统计“输入文本里哪个 pair 最常见”。训练已经决定了
        merge 的优先级。这里每一轮只寻找当前序列中已学过的 pair，并选择最早学到
        的规则（也就是合并后 id 最小的规则）。
        """
        if not isinstance(raw, bytes):
            raise TypeError("raw 必须是 bytes")

        # TODO[BPE-04]: 初始序列为 UTF-8 字节 id；每轮选 rank 最小的已学规则。
        # 本项目合并后 id 即 rank 顺序，不要根据当前输入重新训练。
        token_ids = list(raw)
        while(len(token_ids))>=2:
            pairs = set(zip(token_ids,token_ids[1:]))
            pair = min(pairs, key=lambda p:self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break
            token_ids = self._merge_pair(token_ids,pair,self.merges[pair])
        return token_ids

    def special_id(self, token: str) -> int:
        """查询特殊 token id；分词器尚未训练或名称不存在时明确报错。"""
        try:
            return self.special_tokens[token]
        except KeyError as exc:
            raise ValueError(f"特殊 token 不存在或尚未分配 id: {token!r}") from exc

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """把字符串编码为 token id，可按需在首尾加入 BOS/EOS。"""
        if not isinstance(text, str):
            raise TypeError("text 必须是 str")
        token_ids = self.encode_bytes(text.encode("utf-8"))
        if add_bos:
            token_ids.insert(0, self.special_id("<|bos|>"))
        if add_eos:
            token_ids.append(self.special_id("<|eos|>"))
        return token_ids

    def decode_bytes(
        self, token_ids: Iterable[int], skip_special_tokens: bool = False
    ) -> bytes:
        """把 token id 序列还原为 bytes，可选择跳过特殊 token。"""
        # TODO[BPE-05]: 按 token id 查字节并拼接；特殊 token 可跳过或还原名称。
        # 非整数（含 bool）抛 TypeError，未知 id 抛 ValueError。
        byte_str = list()
        for token in token_ids:
            if isinstance(token,bool) or not isinstance(token,int):
                raise TypeError("token必须是整数")
            vocab_result = self.vocab.get(token)
            special_result = self.id_to_special.get(token)
            if vocab_result == None and special_result!=None:
                if skip_special_tokens:
                    continue
                else:
                    byte_str.append(special_result.encode("utf-8"))
                    continue
            elif vocab_result == None and special_result == None:
                raise ValueError("未知 token id")
            byte_str.append(self.vocab[token])
        return b"".join(byte_str)

        raise NotImplementedError("BPE-05: 请实现字节解码")

    def decode(
        self,
        token_ids: Iterable[int],
        errors: str = "strict",
        skip_special_tokens: bool = False,
    ) -> str:
        """把 token id 序列解码为 UTF-8 字符串。

        正常由 encode 得到的 id 一定可以用 strict 解码。errors 参数主要用于教学：
        若手工传入了只包含半个中文字符的字节 token，可改成 "replace" 观察结果。
        """
        return self.decode_bytes(token_ids, skip_special_tokens).decode(
            "utf-8", errors=errors
        )

    def token_bytes(self, token_id: int) -> bytes:
        """返回单个 token 对应的原始 bytes，便于观察 token 边界。"""
        if token_id in self.id_to_special:
            return self.id_to_special[token_id].encode("utf-8")
        try:
            return self.vocab[token_id]
        except KeyError as exc:
            raise ValueError(f"未知 token id: {token_id}") from exc

    def token_debug(self, token_id: int) -> str:
        """生成同时包含 bytes、十六进制和可显示文本的教学用描述。"""
        raw = self.token_bytes(token_id)
        kind = (
            "special"
            if token_id in self.id_to_special
            else "byte"
            if token_id < self.BYTE_VOCAB_SIZE
            else "merge"
        )
        # 单个 token 可能只是某个多字节 UTF-8 字符的一部分，因此这里不能 strict。
        display = raw.decode("utf-8", errors="replace")
        return (
            f"id={token_id:<6} type={kind:<7} bytes={raw!r:<24} "
            f"hex={raw.hex():<20} text={display!r}"
        )

    def vocabulary_lines(self) -> list[str]:
        """按 id 顺序生成完整词表的可读文本行。"""
        return [self.token_debug(token_id) for token_id in range(self.vocab_size)]

    def print_vocabulary(self) -> None:
        """把完整词表打印到终端。"""
        print(f"词表总大小: {self.vocab_size}")
        print("-" * 100)
        for line in self.vocabulary_lines():
            print(line)

    def save_vocabulary_text(self, path: str | Path) -> None:
        """把与终端相同的完整词表保存为 UTF-8 文本，方便实验报告查阅。"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"词表总大小: {self.vocab_size}", "-" * 100]
        lines.extend(self.vocabulary_lines())
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save(self, path: str | Path) -> None:
        """把词表与 merge 规则保存为便于阅读的 JSON 文件。"""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # JSON 没有 bytes 类型，因此用十六进制字符串无损保存。
        payload = {
            "model_version": self.MODEL_VERSION,
            "type": "byte_level_bpe",
            "special_tokens": self.special_tokens,
            "vocab": {str(token_id): raw.hex() for token_id, raw in self.vocab.items()},
            "merges": [
                {"left": left, "right": right, "new_id": new_id}
                for (left, right), new_id in sorted(
                    self.merges.items(), key=lambda item: item[1]
                )
            ],
        }
        model_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelBPE":
        """从 save 生成的 JSON 文件恢复分词器，并检查模型结构。"""
        model_path = Path(path)
        payload = json.loads(model_path.read_text(encoding="utf-8"))

        if payload.get("model_version") != cls.MODEL_VERSION:
            raise ValueError("不支持的模型版本")
        if payload.get("type") != "byte_level_bpe":
            raise ValueError("模型类型不是 byte_level_bpe")

        tokenizer = cls()
        tokenizer.vocab = {
            int(token_id): bytes.fromhex(hex_value)
            for token_id, hex_value in payload["vocab"].items()
        }
        tokenizer.merges = {
            (rule["left"], rule["right"]): rule["new_id"]
            for rule in payload["merges"]
        }
        tokenizer.special_tokens = {
            str(name): int(token_id)
            for name, token_id in payload.get("special_tokens", {}).items()
        }
        tokenizer.id_to_special = {
            token_id: name for name, token_id in tokenizer.special_tokens.items()
        }
        tokenizer._validate_model()
        return tokenizer

    def _validate_model(self) -> None:
        """检查载入模型的关键不变量，避免损坏的 JSON 产生悄悄的错误。"""
        for byte_value in range(256):
            if self.vocab.get(byte_value) != bytes([byte_value]):
                raise ValueError(f"基础字节 token {byte_value} 缺失或内容错误")

        expected_new_ids = list(range(256, 256 + len(self.merges)))
        actual_new_ids = sorted(self.merges.values())
        if actual_new_ids != expected_new_ids:
            raise ValueError("merge 的新 token id 必须从 256 开始连续递增")

        for (left, right), new_id in self.merges.items():
            if left not in self.vocab or right not in self.vocab:
                raise ValueError(f"merge {new_id} 引用了词表中不存在的 token")
            if left >= new_id or right >= new_id:
                raise ValueError("merge 只能引用在它之前已经存在的 token")
            expected_bytes = self.vocab[left] + self.vocab[right]
            if self.vocab.get(new_id) != expected_bytes:
                raise ValueError(f"token {new_id} 的字节内容与 merge 规则不一致")

        if self.special_tokens:
            expected_special_ids = list(
                range(len(self.vocab), len(self.vocab) + len(self.SPECIAL_TOKENS))
            )
            actual_special_ids = [
                self.special_tokens.get(name) for name in self.SPECIAL_TOKENS
            ]
            if actual_special_ids != expected_special_ids:
                raise ValueError("特殊 token id 必须按配置顺序紧跟普通 BPE 词表")
        expected_reverse_mapping = {
            token_id: name for name, token_id in self.special_tokens.items()
        }
        if self.id_to_special != expected_reverse_mapping:
            raise ValueError("特殊 token 的正向与反向映射不一致")

    @property
    def vocab_size(self) -> int:
        """返回实际词表大小；短语料可能无法达到请求的目标大小。"""
        return len(self.vocab) + len(self.special_tokens)
