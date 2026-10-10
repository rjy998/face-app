"""
大模型连通性体检：确认当前配置的 provider 能不能正常出文本。

用法（在 backend 目录下）：
    python scripts/check_llm.py                      # 用 .env 里的配置发一次最小请求
    python scripts/check_llm.py --models A B C       # 依次试探多个模型名（找可用模型时用）
    python scripts/check_llm.py --provider anthropic --base-url https://api.minimax.cn/anthropic --key sk-xxx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from config import settings  # noqa: E402
from services import interpreter  # noqa: E402


def probe(provider: dict[str, str], question: str = "用一句话说明什么是三庭五眼。") -> tuple[bool, str, float]:
    messages = [
        {"role": "system", "content": "你是面相文化顾问，回答要简短、温和。"},
        {"role": "user", "content": question},
    ]
    started = time.time()
    try:
        text = interpreter._chat_completion(provider, messages)  # noqa: SLF001 - 体检脚本直接用内部实现
        return True, text, time.time() - started
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}", time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="大模型连通性体检")
    parser.add_argument("--provider", default=None, help="临时覆盖 LLM_PROVIDER（doubao/openai/anthropic）")
    parser.add_argument("--base-url", default=None, help="临时覆盖 base_url")
    parser.add_argument("--key", default=None, help="临时覆盖 API Key")
    parser.add_argument("--models", nargs="*", default=None, help="依次试探的模型名")
    args = parser.parse_args()

    if args.provider:
        settings.LLM_PROVIDER = args.provider
    if args.base_url:
        if (args.provider or settings.LLM_PROVIDER) in {"anthropic", "minimax"}:
            settings.ANTHROPIC_BASE_URL = args.base_url
        else:
            settings.OPENAI_BASE_URL = args.base_url
    if args.key:
        if (args.provider or settings.LLM_PROVIDER) in {"anthropic", "minimax"}:
            settings.ANTHROPIC_API_KEY = args.key
        else:
            settings.OPENAI_API_KEY = args.key

    resolved = interpreter._resolve_provider()  # noqa: SLF001
    if resolved is None:
        print("没有配置任何大模型密钥：当前会使用本地模板生成解读。")
        print("请在 backend/.env 里填写 ANTHROPIC_API_KEY / ARK_API_KEY / OPENAI_API_KEY 之一。")
        return 1

    print(f"provider={resolved['name']} 协议={resolved['protocol']}")
    print(f"base_url={resolved['base_url']}")
    print(f"model={resolved['model']}")

    models = args.models or [resolved["model"]]
    ok_any = False
    for model in models:
        candidate = dict(resolved, model=model)
        ok, text, cost = probe(candidate)
        if ok:
            ok_any = True
            print(f"\n[OK] 模型 {model}（{cost:.1f}s）")
            print(text[:300])
        else:
            print(f"\n[FAIL] 模型 {model}（{cost:.1f}s）")
            print("   ", text[:500])

    return 0 if ok_any else 2


if __name__ == "__main__":
    raise SystemExit(main())
