# 当前训练语料：corpus_en.txt 与 corpus_zh.txt

项目把中英文 corpus 分开存储，训练开始时才在内存中合并。

## 文件结构

```text
data/
├── corpus_en.txt   # 英文语料
├── corpus_zh.txt   # 中文语料
└── README.md       # 本说明
```

## 文件统计

| 文件 | 行数 | 字节数 | SHA-256 |
|---|---:|---:|---|
| `corpus_en.txt` | 10,000 | 614,986 | `11ff800a8fb1e0aefecd16494321ad830bd4f1c27734f648be0698caa3c81d2e` |
| `corpus_zh.txt` | 10,000 | 548,828 | `93ac7fc5fa8b0e9fde4f4849562a76a46c81b6d4c122b1c66cc1f739c5bfbea0` |

两个文件都是 UTF-8 文本，相同行号互为翻译。例如第一行：

```text
corpus_en.txt: Do you think we look young enough to blend in at a high school?
corpus_zh.txt: 你们觉得我们看起来够年轻溜进高中吗？
```

原始行首尾包含空格，训练代码会保留这些空格，因为空格同样属于可学习的字节。文件使用
Windows `CRLF` 换行。Python 文本模式会把 `CRLF` 统一为 `LF`；合并前会去除末尾
换行，再显式插入一个 `\n`，避免跨文件粘连。

两个磁盘文件共 1,163,814 字节，规范化换行后实际送入训练的是 1,143,816 个 UTF-8
字节。

## 训练时如何合并

[demo_bpe.py](../demo_bpe.py) 使用以下逻辑：

```python
english_corpus = Path("data/corpus_en.txt").read_text(encoding="utf-8")
chinese_corpus = Path("data/corpus_zh.txt").read_text(encoding="utf-8")

training_text = (
    english_corpus.rstrip("\r\n")
    + "\n"
    + chinese_corpus.rstrip("\r\n")
    + "\n"
)

tokenizer.train(training_text, vocab_size=1024, show_progress=True)
```

命令行训练：

```powershell
python bpe_cli.py train `
  --corpus data/corpus_en.txt data/corpus_zh.txt `
  --vocab-size 1024 `
  --model bpe_model.json
```

`bpe_cli.py` 会按参数顺序读取文件，在相邻文件之间补一个换行，然后送入同一个 BPE
训练过程。1024 是包含 4 个特殊 token 在内的最终总词表大小，因此实际最多学习
`1024 - 256 - 4 = 764` 条 merge 规则；命令默认显示训练进度条。

