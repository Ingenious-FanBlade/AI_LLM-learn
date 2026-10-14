"""ByteLevelBPE 的自动化测试。

运行：python -m pytest -v
每个测试对应 README 中的一个验收点。
"""

import json

import pytest

from lab.implementation import get_bpe_class

ByteLevelBPE = get_bpe_class()


@pytest.fixture
def trained_bpe() -> ByteLevelBPE:
    """给多个测试复用的小模型；每个测试会得到独立实例。"""
    tokenizer = ByteLevelBPE()
    tokenizer.train(
        "hello hello hello 你好世界 人工智能 AI 🤖 hello 你好",
        vocab_size=290,
    )
    return tokenizer


def test_initial_vocab_contains_every_byte() -> None:
    """字节级 BPE 初始词表必须无损覆盖 0~255。"""
    tokenizer = ByteLevelBPE()
    assert tokenizer.vocab_size == 256
    assert tokenizer.special_tokens == {}  # 特殊 token id 在训练结束后分配。
    assert all(tokenizer.vocab[i] == bytes([i]) for i in range(256))


def test_training_learns_merges_and_shrinks_training_sequence() -> None:
    """重复语料应产生 merge，编码后的 token 数应少于原始字节数。"""
    text = "low low low lower"
    tokenizer = ByteLevelBPE()
    tokenizer.train(text, vocab_size=270)
    encoded = tokenizer.encode(text)

    assert tokenizer.merges
    assert len(encoded) < len(text.encode("utf-8"))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello",
        "hello hello hello",
        "aaaaaaaaaaaa",
        "你好世界",
        "人工智能 AI",
        "Hello, 世界!",
        "🤖🔥😀",
        "换行也可以\n第二行\t制表符",
    ],
)
def test_round_trip_for_multilingual_and_edge_cases(
    trained_bpe: ByteLevelBPE, text: str
) -> None:
    """最重要的不变量：decode(encode(text)) 必须严格等于原文。"""
    assert trained_bpe.decode(trained_bpe.encode(text)) == text


def test_encode_does_not_learn_new_pairs(trained_bpe: ByteLevelBPE) -> None:
    """encode 只能应用旧规则，不能根据输入频率临时训练新规则。"""
    before_merges = trained_bpe.merges.copy()
    ids = trained_bpe.encode("zzzzzzzzzz")

    assert trained_bpe.merges == before_merges
    # 训练语料没有 z，因此 (z, z) 不会被合并。
    assert ids == [ord("z")] * 10


def test_overlapping_pair_is_merged_from_left_to_right() -> None:
    """三个相同 token 中只能合并一个不重叠 pair。"""
    assert ByteLevelBPE._merge_pair([1, 1, 1], (1, 1), 256) == [256, 1]


def test_tie_break_is_deterministic() -> None:
    """ab 与 bc 都出现一次时，字典序较小的 (a,b) 先被选择。"""
    tokenizer = ByteLevelBPE()
    # 256 基础字节 + 1 个 merge + 4 个特殊 token = 261。
    tokenizer.train("abc", vocab_size=261)
    assert tokenizer.merges == {(ord("a"), ord("b")): 256}


def test_save_and_load_preserve_behavior(tmp_path) -> None:
    """模型落盘后，词表、规则和编码结果必须完全一致。"""
    original = ByteLevelBPE()
    original.train("banana banana 你好你好", vocab_size=275)
    model_path = tmp_path / "model.json"
    original.save(model_path)
    loaded = ByteLevelBPE.load(model_path)

    text = "banana 与你好 🤖"
    assert loaded.vocab == original.vocab
    assert loaded.merges == original.merges
    assert loaded.encode(text) == original.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_load_rejects_broken_base_vocab(tmp_path) -> None:
    """损坏的模型应明确报错，而不是悄悄给出错误编码。"""
    tokenizer = ByteLevelBPE()
    model_path = tmp_path / "broken.json"
    tokenizer.save(model_path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    payload["vocab"]["65"] = "00"
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="基础字节 token 65"):
        ByteLevelBPE.load(model_path)


def test_invalid_vocab_size_is_rejected() -> None:
    """总词表必须容纳 256 个基础字节和 4 个特殊 token。"""
    tokenizer = ByteLevelBPE()
    with pytest.raises(ValueError, match="不能小于 260"):
        tokenizer.train("hello", vocab_size=259)


def test_unknown_token_id_is_rejected(trained_bpe: ByteLevelBPE) -> None:
    """decode 收到模型中不存在的 id 时应给出清晰异常。"""
    with pytest.raises(ValueError, match="未知 token id"):
        trained_bpe.decode([999_999])


def test_tiktoken_is_also_reversible() -> None:
    """参考实现测试：这里只比可逆性，不要求 token id 与手写模型相同。"""
    tiktoken = pytest.importorskip("tiktoken")
    try:
        reference = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        # 首次载入 cl100k_base 需要下载官方词表；离线环境不应让核心 BPE 测试失败。
        pytest.skip(f"tiktoken 官方词表尚未缓存且当前不可下载: {exc}")
    text = "你好，Artificial Intelligence! 🤖"
    assert reference.decode(reference.encode(text)) == text


def test_special_tokens_are_inside_target_vocab_and_can_be_skipped() -> None:
    """特殊 token 计入总大小，BOS/EOS 可添加且解码时可跳过。"""
    tokenizer = ByteLevelBPE()
    tokenizer.train(
        "banana bandana cabana banana bandana cabana", vocab_size=268
    )

    assert tokenizer.vocab_size == 268
    assert tokenizer.special_tokens == {
        "<|pad|>": 264,
        "<|bos|>": 265,
        "<|eos|>": 266,
        "<|unk|>": 267,
    }

    ids = tokenizer.encode("banana", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.special_id("<|bos|>")
    assert ids[-1] == tokenizer.special_id("<|eos|>")
    assert tokenizer.decode(ids) == "<|bos|>banana<|eos|>"
    assert tokenizer.decode(ids, skip_special_tokens=True) == "banana"


def test_training_progress_reaches_100_percent(capsys) -> None:
    """训练进度条必须能够到达 100%。"""
    tokenizer = ByteLevelBPE()
    tokenizer.train("abababab abab", vocab_size=263, show_progress=True)
    output = capsys.readouterr().out

    assert "训练进度" in output
    assert "100.0%" in output


def test_vocabulary_lines_include_every_id_and_special_token() -> None:
    """最终词表打印内容必须覆盖所有 id 和四个特殊 token。"""
    tokenizer = ByteLevelBPE()
    tokenizer.train("hello hello hello", vocab_size=264)
    lines = tokenizer.vocabulary_lines()

    assert len(lines) == tokenizer.vocab_size == 264
    assert lines[0].startswith("id=0")
    assert "type=special" in lines[-1]
    assert "<|unk|>" in lines[-1]
