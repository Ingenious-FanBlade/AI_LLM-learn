"""实验统一入口；默认运行学生实现。"""
import argparse
import importlib
import os
import sys

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="实验二：BPE 与自注意力")
    parser.add_argument("--implementation", choices=["student", "reference"], default="student")
    parser.add_argument("task", choices=["bpe-demo", "attention-demo", "bpe-cli"])
    args, remaining = parser.parse_known_args()
    if args.task != "bpe-cli" and remaining:
        parser.error("演示命令不接受额外参数")
    os.environ["LAB_IMPLEMENTATION"] = args.implementation
    print(f"当前实现: lab.{args.implementation}", flush=True)
    module = {"bpe-demo": "lab.demos.bpe", "attention-demo": "lab.demos.attention",
              "bpe-cli": "lab.cli.bpe"}[args.task]
    sys.argv = [sys.argv[0]] + remaining
    importlib.import_module(module).main()

if __name__ == "__main__":
    main()
