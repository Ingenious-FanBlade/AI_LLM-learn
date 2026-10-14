"""统一选择实现，不将教师答案导入学生代码。"""

import importlib
import os


def get_implementation_name() -> str:
    name = os.environ.get("LAB_IMPLEMENTATION", "student")
    if name not in {"student", "reference"}:
        raise ValueError("LAB_IMPLEMENTATION 只能是 student 或 reference")
    return name


def get_bpe_class():
    return importlib.import_module(
        f"lab.{get_implementation_name()}.bpe"
    ).ByteLevelBPE


def get_attention_module():
    return importlib.import_module(f"lab.{get_implementation_name()}.attention")
