"""字节级 BPE 的命令行工具。

先运行 ``python bpe_cli.py --help`` 查看总帮助，再运行某个子命令的 ``--help``。
这个脚本把训练、编码、解码和 tiktoken 对比拆开，便于逐步验收实验。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lab.implementation import get_bpe_class

ByteLevelBPE = get_bpe_class()


def _configure_console_encoding() -> None:
    """避免 Windows GBK 控制台打印 emoji 时触发 UnicodeEncodeError。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def _print_tokens(tokenizer: ByteLevelBPE, token_ids: list[int]) -> None:
    """逐行打印 token id 与它对应的底层字节。"""
    if not token_ids:
        print("  （空 token 序列）")
        return
    for token_id in token_ids:
        print(" ", tokenizer.token_debug(token_id))


def command_train(args: argparse.Namespace) -> None:
    """处理 train 子命令。"""
    corpus_parts = [Path(path).read_text(encoding="utf-8") for path in args.corpus]
    # 支持一个或多个语料文件。每部分之间补一个换行，防止前一文件末尾没有换行时
    # 与后一文件开头意外形成跨文件 token pair。
    corpus = "\n".join(part.rstrip("\n") for part in corpus_parts) + "\n"
    tokenizer = ByteLevelBPE()

    print("训练语料（按以下顺序在内存中合并）:")
    for path, part in zip(args.corpus, corpus_parts):
        print(f"  {path}: {len(part.encode('utf-8')):,} UTF-8 字节")
    print(f"合并后 UTF-8 字节数: {len(corpus.encode('utf-8')):,}")
    print(f"目标词表大小: {args.vocab_size}")
    tokenizer.train(
        corpus,
        args.vocab_size,
        show_progress=args.progress,
    )
    tokenizer.save(args.model)

    print(f"实际词表大小: {tokenizer.vocab_size}")
    print(f"学到的 merge 数: {len(tokenizer.merges)}")
    print(f"特殊 token: {tokenizer.special_tokens}")
    print(f"模型已保存: {args.model}")


def command_encode(args: argparse.Namespace) -> None:
    """处理 encode 子命令。"""
    tokenizer = ByteLevelBPE.load(args.model)
    token_ids = tokenizer.encode(
        args.text, add_bos=args.add_bos, add_eos=args.add_eos
    )

    print("token ids:")
    print(token_ids)
    print("token 明细:")
    _print_tokens(tokenizer, token_ids)
    print(f"原始 UTF-8 字节数: {len(args.text.encode('utf-8'))}")
    print(f"编码后的 token 数: {len(token_ids)}")


def command_decode(args: argparse.Namespace) -> None:
    """处理 decode 子命令。"""
    tokenizer = ByteLevelBPE.load(args.model)
    print(tokenizer.decode(args.ids, skip_special_tokens=args.skip_special_tokens))


def command_vocab(args: argparse.Namespace) -> None:
    """打印模型的完整词表，并可额外保存为文本文件。"""
    tokenizer = ByteLevelBPE.load(args.model)
    if args.output:
        tokenizer.save_vocabulary_text(args.output)
        print(f"词表已保存: {args.output}")
    tokenizer.print_vocabulary()


def command_compare(args: argparse.Namespace) -> None:
    """把手写 BPE 与 tiktoken 放在相同文本上进行可逆性和粒度对比。"""
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit(
            "compare 需要 tiktoken，请先运行: python -m pip install -r requirements.txt"
        ) from exc

    tokenizer = ByteLevelBPE.load(args.model)
    try:
        # 第一次使用该 encoding 时，tiktoken 可能需要联网下载官方词表。
        reference = tiktoken.get_encoding(args.encoding)
    except Exception as exc:
        raise SystemExit(
            f"无法载入 tiktoken encoding {args.encoding!r}。"
            f"首次运行请保持联网。原始错误: {exc}"
        ) from exc

    my_ids = tokenizer.encode(args.text)
    reference_ids = reference.encode(args.text)

    print("=== 手写 ByteLevelBPE ===")
    print("ids:", my_ids)
    _print_tokens(tokenizer, my_ids)
    print("decode 正确:", tokenizer.decode(my_ids) == args.text)

    print(f"\n=== tiktoken / {args.encoding} ===")
    print("ids:", reference_ids)
    for token_id in reference_ids:
        raw = reference.decode_single_token_bytes(token_id)
        display = raw.decode("utf-8", errors="replace")
        print(
            f"  id={token_id:<6} bytes={raw!r:<24} "
            f"hex={raw.hex():<20} text={display!r}"
        )
    print("decode 正确:", reference.decode(reference_ids) == args.text)

    print("\n=== 统计结论 ===")
    print(f"原始 UTF-8 字节数: {len(args.text.encode('utf-8'))}")
    print(f"手写 BPE token 数: {len(my_ids)}")
    print(f"tiktoken token 数: {len(reference_ids)}")
    print("注意：两套模型的 token id 没有可比性，重点应比较可逆性、数量和边界。")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="从零实现的字节级 BPE：训练、编码、解码和 tiktoken 对比"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="从 UTF-8 文本训练 BPE 模型")
    train_parser.add_argument(
        "--corpus",
        required=True,
        nargs="+",
        help="一个或多个训练语料文件；多个文件会按顺序在内存中合并",
    )
    train_parser.add_argument(
        "--vocab-size",
        type=int,
        default=1024,
        help="最终总词表大小（包含特殊 token），默认 1024",
    )
    train_parser.add_argument("--model", required=True, help="输出的 JSON 模型路径")
    train_parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="关闭实时训练进度条",
    )
    train_parser.set_defaults(progress=True)
    train_parser.set_defaults(func=command_train)

    encode_parser = subparsers.add_parser("encode", help="用已经训练的模型编码文本")
    encode_parser.add_argument("--model", required=True, help="JSON 模型路径")
    encode_parser.add_argument("--text", required=True, help="要编码的文本")
    encode_parser.add_argument("--add-bos", action="store_true", help="在开头加入 BOS")
    encode_parser.add_argument("--add-eos", action="store_true", help="在结尾加入 EOS")
    encode_parser.set_defaults(func=command_encode)

    decode_parser = subparsers.add_parser("decode", help="把 token ids 解码为文本")
    decode_parser.add_argument("--model", required=True, help="JSON 模型路径")
    decode_parser.add_argument("ids", type=int, nargs="*", help="一个或多个 token id")
    decode_parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="解码时不输出 PAD/BOS/EOS/UNK 文本标记",
    )
    decode_parser.set_defaults(func=command_decode)

    vocab_parser = subparsers.add_parser("vocab", help="打印完整词表")
    vocab_parser.add_argument("--model", required=True, help="JSON 模型路径")
    vocab_parser.add_argument("--output", help="可选：同时保存为 UTF-8 文本")
    vocab_parser.set_defaults(func=command_vocab)

    compare_parser = subparsers.add_parser("compare", help="与 tiktoken 对比")
    compare_parser.add_argument("--model", required=True, help="JSON 模型路径")
    compare_parser.add_argument("--text", required=True, help="两边共同编码的文本")
    compare_parser.add_argument(
        "--encoding", default="cl100k_base", help="tiktoken 编码名称"
    )
    compare_parser.set_defaults(func=command_compare)
    return parser


def main() -> None:
    """命令行入口。"""
    _configure_console_encoding()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
