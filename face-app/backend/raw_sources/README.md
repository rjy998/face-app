# raw_sources —— 原始资料投放目录

把**未加工的原始资料**丢到这里，然后跑一条命令，剩下的（抽文字 → 大模型整理 → 入库）全自动。

## 支持的文件

| 类型 | 说明 |
|---|---|
| `.pdf` 数字版 | 文字可以直接选中复制 → 用 PyMuPDF 直接抽取文字层 |
| `.pdf` 扫描版 | 整页是图片（古籍影印、扫描件）→ 自动栅格化后用中文 OCR 逐页识别 |
| `.pdf` 混合版 | 部分页有文字层、部分页是图 → **逐页判断**，有文字层抽文字、没有的走 OCR |
| `.txt` / `.md` | 已经整理好的纯文本，直接进入「大模型整理」环节 |

繁体、竖排的古籍都可以直接放进来：竖排页面会按「从右到左、每列从上到下」重排阅读顺序。

## 用法

```bash
cd backend

# 一键：抽取 → 整理 → 入库
python scripts/auto_build_kb.py --all

# 分步执行（便于检查中间结果）
python scripts/auto_build_kb.py --extract      # PDF → raw_sources/extracted/*.txt
python scripts/auto_build_kb.py --distill      # txt  → knowledge_base/docs/*.md
python scripts/auto_build_kb.py --ingest       # md   → 向量库

# 只处理部分页、调整识别精度
python scripts/auto_build_kb.py --extract --pages 1-30 --dpi 250

# 指定 OCR 引擎（默认自动挑：paddleocr → rapidocr）
python scripts/auto_build_kb.py --extract --ocr-engine rapidocr
```

## 中间产物放在哪

| 目录 | 内容 | 要不要提交到 Git |
|---|---|---|
| `raw_sources/` | 你放进来的原始 PDF | **不建议**（体积大、可能涉版权） |
| `raw_sources/extracted/` | 抽出来的原文 txt（中间产物，可重复生成） | 不建议 |
| `knowledge_base/docs/` | 大模型整理后的 Markdown（真正入库的内容） | 建议自己留存 |

## 关于版权与合规

- 古籍影印件、扫描件请注意来源与版权，公开部署前确认授权；
- 整理环节会自动把「必定发财」「克夫」「大凶」等绝对化、恐吓性表述改写成中性的
  「传统相学认为……，供文化参考」，但**你仍应通读一遍**再对外使用；
- 原始 PDF 只在你的机器上处理；只有**抽出来的文字**会发给大模型做整理。
  如需完全离线，`.env` 里把 `LLM_PROVIDER` 设为 `template`（此时不会调用任何外部接口，
  但也就不做整理了，脚本会提示你改用本地模型或手动整理）。
