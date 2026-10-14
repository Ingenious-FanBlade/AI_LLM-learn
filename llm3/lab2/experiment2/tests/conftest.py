"""同一套公开测试可验收参考答案或学生实现。"""

import os


def pytest_addoption(parser):
    parser.addoption(
        "--implementation",
        choices=["student", "reference"],
        default=os.environ.get("LAB_IMPLEMENTATION", "student"),
        help="选择待验收的实现，默认 student",
    )


def pytest_configure(config):
    os.environ["LAB_IMPLEMENTATION"] = config.getoption("--implementation")


def pytest_report_header(config):
    return f"实验二验收对象: lab.{config.getoption('--implementation')}"
