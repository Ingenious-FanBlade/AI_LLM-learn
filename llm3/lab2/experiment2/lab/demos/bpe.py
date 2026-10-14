"""一键运行完整的字节级 BPE 实验。

运行方式：python demo_bpe.py
它会依次演示 UTF-8 字节、训练、编解码、保存/加载和 tiktoken 对比。
"""

from __future__ import annotations

import sys
from pathlib import Path

from lab.implementation import get_bpe_class

ByteLevelBPE = get_bpe_class()


PROJECT_DIR = Path(__file__).resolve().parents[2]
ENGLISH_CORPUS_PATH = PROJECT_DIR / "data" / "corpus_en.txt"
CHINESE_CORPUS_PATH = PROJECT_DIR / "data" / "corpus_zh.txt"
MODEL_PATH = PROJECT_DIR / "outputs" / "bpe" / "bpe_model.json"
VOCAB_PATH = PROJECT_DIR / "outputs" / "bpe" / "vocab.txt"
TARGET_VOCAB_SIZE = 512


def configure_console_encoding() -> None:
    """让 Windows 的旧 GBK 终端也能稳定打印中文和 emoji。

    reconfigure 只改变当前 Python 进程的标准输出，不修改用户的系统设置。
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def show_utf8_basics() -> None:
    """步骤 1：观察 Unicode 字符串在底层对应的 UTF-8 字节。"""
    text = "你好A🤖"
    raw = text.encode("utf-8")
    print("\n[步骤 1] 字符串 -> UTF-8 bytes")
    print("文本:", text)
    print("bytes:", raw)
    print("每个字节的十进制值:", list(raw))
    print("说明：A 占 1 字节，中文通常占 3 字节，🤖 占 4 字节。")


def train_model() -> ByteLevelBPE:
    """步骤 2：把 corpus_en/corpus_zh 合并为训练语料并训练 BPE。"""
    english_corpus = ENGLISH_CORPUS_PATH.read_text(encoding="utf-8")
    chinese_corpus = CHINESE_CORPUS_PATH.read_text(encoding="utf-8")

    # 两个 corpus 文件在磁盘上分开保存，训练时才在内存中合并。
    # 当前文件末尾没有换行，因此先删除可能存在的尾部换行，再显式补一个分隔换行。
    # 这样最后一条英文和第一条中文不会粘在一起，也不会引入多余空行。
    corpus = (
        english_corpus.rstrip("\r\n")
        + "\n"
        + chinese_corpus.rstrip("\r\n")
        + "\n"
    )
    tokenizer = ByteLevelBPE()

    print("\n[步骤 2] 训练 BPE")
    print(
        f"英文语料 corpus_en.txt: {len(english_corpus.splitlines()):,} 行，"
        f"{len(english_corpus.encode('utf-8')):,} 字节"
    )
    print(
        f"中文语料 corpus_zh.txt: {len(chinese_corpus.splitlines()):,} 行，"
        f"{len(chinese_corpus.encode('utf-8')):,} 字节"
    )
    print(f"训练时合并: {len(corpus.encode('utf-8')):,} UTF-8 字节")
    print(
        f"词表组成: 256 个基础字节 + {TARGET_VOCAB_SIZE - 260} 个 merge "
        f"+ 4 个特殊 token = {TARGET_VOCAB_SIZE}"
    )
    print("训练过程将实时显示进度、pair 频次、当前序列长度和预计剩余时间。")
    tokenizer.train(corpus, vocab_size=TARGET_VOCAB_SIZE, show_progress=True)
    print(
        f"训练完成：vocab={tokenizer.vocab_size}, merges={len(tokenizer.merges)}, "
        f"special={len(tokenizer.special_tokens)}"
    )
    print("特殊 token:", tokenizer.special_tokens)

    # 只统计能独立解成 UTF-8、且包含中日韩统一表意文字的“新 token”。
    # 这能直接证明模型确实从双语语料中学到了中文片段，而不只是逐字节兜底。
    learned_chinese_tokens: list[tuple[int, str]] = []
    for token_id in range(256, tokenizer.vocab_size):
        try:
            piece = tokenizer.token_bytes(token_id).decode("utf-8")
        except UnicodeDecodeError:
            continue
        if any("\u4e00" <= character <= "\u9fff" for character in piece):
            learned_chinese_tokens.append((token_id, piece))

    assert learned_chinese_tokens, "双语语料训练后应该至少学到一个完整中文 token"
    print("学到的部分中文 token:", learned_chinese_tokens[:8])
    return tokenizer


def verify_round_trip(tokenizer: ByteLevelBPE) -> None:
    """步骤 3：对多种语言和边界情况验证 encode/decode 可逆。"""
    samples = [
        "hello hello hello",
        "你好世界",
        "Hello, 世界!",
        "🤖🔥😀",
    ]

    print("\n[步骤 3] 编码、解码与可逆性检查")
    for text in samples:
        token_ids = tokenizer.encode(text)
        recovered = tokenizer.decode(token_ids)
        assert recovered == text
        print(f"文本={text!r}")
        print(f"  ids={token_ids}")
        print(f"  token 数={len(token_ids)}, round-trip=True")

    # BOS 表示序列开始，EOS 表示序列结束。解码时可保留标记用于调试，也可跳过它们
    # 恢复纯正文。PAD 用于把同一 batch 中的序列补到等长；UNK 在字节级 BPE 中
    # 通常不会自动出现，因为所有 UTF-8 输入都能由 0~255 基础字节表示。
    special_example = "你好, BPE!"
    special_ids = tokenizer.encode(special_example, add_bos=True, add_eos=True)
    assert special_ids[0] == tokenizer.special_id("<|bos|>")
    assert special_ids[-1] == tokenizer.special_id("<|eos|>")
    assert tokenizer.decode(special_ids, skip_special_tokens=True) == special_example
    print("特殊 token 编码示例:", special_ids)
    print("保留特殊 token 解码:", tokenizer.decode(special_ids))
    print(
        "跳过特殊 token 解码:",
        tokenizer.decode(special_ids, skip_special_tokens=True),
    )


def save_and_reload(tokenizer: ByteLevelBPE) -> ByteLevelBPE:
    """步骤 4：持久化模型，证明训练和使用可以分成两个进程。"""
    print("\n[步骤 4] 保存并重新加载模型")
    tokenizer.save(MODEL_PATH)
    loaded = ByteLevelBPE.load(MODEL_PATH)
    probe = "保存后的模型仍然可以编码 🤖"
    assert loaded.decode(loaded.encode(probe)) == probe
    print("模型路径:", MODEL_PATH)
    print("重新加载后的检查: passed")
    return loaded


def compare_with_tiktoken(tokenizer: ByteLevelBPE) -> None:
    """步骤 5：与 tiktoken 比较 token 数量、边界和可逆性。"""
    try:
        import tiktoken
    except ImportError:
        print("\n[步骤 5] 未安装 tiktoken，跳过；请执行 pip install -r requirements.txt")
        return

    text = "Artificial intelligence 与机器学习很有趣 🤖"
    try:
        # tiktoken 第一次使用某个 encoding 时可能需要联网下载词表并写入缓存。
        reference = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        print("\n[步骤 5] tiktoken 词表暂时不可用，跳过本次对比。")
        print("首次运行通常需要联网下载 cl100k_base；错误:", exc)
        return
    my_ids = tokenizer.encode(text)
    tk_ids = reference.encode(text)

    print("\n[步骤 5] 与 tiktoken / cl100k_base 对比")
    print("文本:", text)
    print("手写 BPE ids:", my_ids)
    print("tiktoken ids:", tk_ids)
    print("手写 BPE token 数:", len(my_ids))
    print("tiktoken token 数:", len(tk_ids))
    print("手写 BPE round-trip:", tokenizer.decode(my_ids) == text)
    print("tiktoken round-trip:", reference.decode(tk_ids) == text)
    print("说明：两者训练语料、词表、预分词规则不同，ids 不同是正常现象。")


def print_and_save_vocabulary(tokenizer: ByteLevelBPE) -> None:
    """步骤 6：保存并在终端打印实际完整词表。"""
    print("\n[步骤 6] 完整词表")
    tokenizer.save_vocabulary_text(VOCAB_PATH)
    print("词表文本已保存:", VOCAB_PATH)
    tokenizer.print_vocabulary()


def main() -> None:
    """按教学顺序执行整套实验。"""
    configure_console_encoding()
    print("=" * 64)
    print("手写字节级 BPE 实验：训练 / 编码 / 解码 / tiktoken 对比")
    print("=" * 64)
    show_utf8_basics()
    tokenizer = train_model()
    verify_round_trip(tokenizer)
    tokenizer = save_and_reload(tokenizer)
    compare_with_tiktoken(tokenizer)
    print_and_save_vocabulary(tokenizer)
    print("\n全部步骤执行完成。")


if __name__ == "__main__":
    main()
