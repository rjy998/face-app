"""
本地批量导入知识库（不经过 HTTP，直接入库）。

用法（在 backend 目录下执行）：
    python scripts/ingest_local.py                     # 导入 knowledge_base/docs/ 下全部 md/txt
    python scripts/ingest_local.py --path ../我的资料    # 导入指定目录
    python scripts/ingest_local.py --file 相学.md        # 导入单个文件
    python scripts/ingest_local.py --list               # 只看当前知识库统计
    python scripts/ingest_local.py --reset              # 清空知识库后重新导入
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本能直接 import backend 下的模块
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from config import KB_DOCS_DIR  # noqa: E402
from models.db import KbChunk, KbDocument, SessionLocal, init_db  # noqa: E402
from services import rag  # noqa: E402


def reset_kb() -> None:
    with SessionLocal() as db:
        deleted_chunks = db.query(KbChunk).delete()
        deleted_docs = db.query(KbDocument).delete()
        db.commit()
    print(f"已清空知识库：删除文档 {deleted_docs} 篇、切片 {deleted_chunks} 条")


def main() -> int:
    parser = argparse.ArgumentParser(description="批量导入面相知识库")
    parser.add_argument("--path", default=str(KB_DOCS_DIR), help="要导入的目录（递归处理 .md/.txt）")
    parser.add_argument("--file", default=None, help="只导入单个文件")
    parser.add_argument("--source", default=None, help="来源说明，如书名或链接")
    parser.add_argument("--list", action="store_true", help="只打印知识库统计")
    parser.add_argument("--reset", action="store_true", help="先清空知识库再导入")
    args = parser.parse_args()

    init_db()
    if args.reset:
        reset_kb()

    with SessionLocal() as db:
        if args.list:
            print("知识库统计：", rag.stats(db))
            return 0

        if args.file:
            outcome = rag.ingest_file(db, args.file, source=args.source)
            print("导入结果：", outcome)
        else:
            result = rag.ingest_directory(db, args.path)
            print("导入结果：", result)
        print("知识库统计：", rag.stats(db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
