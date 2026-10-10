"""
端到端自测脚本（只依赖 Python 标准库，直接打真实 HTTP 接口）。

用法：
    1. 先启动后端：uvicorn main:app --port 8000
    2. 另开一个终端：python scripts/selftest.py            # 默认 http://127.0.0.1:8000
       python scripts/selftest.py --base http://127.0.0.1:8000 --keep

覆盖用例：
    /api/health → /api/auth/guest → /api/face/analyze（真实上传图片）
    → /api/history → /api/history/{id} → /api/kb/query → /api/face/image/{id}
    → /api/history/{id} 删除 → 删除后 404
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = BACKEND_DIR / "storage" / "testdata" / "test_face.jpg"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}{(' — ' + detail) if detail else ''}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}{(' — ' + detail) if detail else ''}")
    return condition


def http(method: str, url: str, *, token: str | None = None, body: bytes | None = None,
         content_type: str | None = None, timeout: int = 120):
    """返回 (status_code, 解析后的 body 或 bytes, headers)。"""
    request = urllib.request.Request(url, data=body, method=method)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if content_type:
        request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            meta = response.headers          # HTTPMessage：取值大小写不敏感
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read()
        meta = error.headers
        status = error.code

    ctype = meta.get("Content-Type", "") or ""
    if "application/json" in ctype.lower():
        try:
            return status, json.loads(raw.decode("utf-8")), meta
        except json.JSONDecodeError:
            return status, raw, meta
    return status, raw, meta


def post_json(url: str, payload: dict, token: str | None = None):
    return http("POST", url, token=token, body=json.dumps(payload).encode("utf-8"),
                content_type="application/json")


def post_file(url: str, path: Path, token: str, field: str = "file"):
    boundary = f"----selftest{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    chunks: list[bytes] = []
    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(
        f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'.encode()
    )
    chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode())
    chunks.append(path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return http(
        "POST", url, token=token, body=b"".join(chunks),
        content_type=f"multipart/form-data; boundary={boundary}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="面相应用端到端自测")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="后端地址")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="用于分析的测试图片")
    parser.add_argument("--keep", action="store_true", help="保留测试记录（默认测完即删）")
    parser.add_argument("--verbose", action="store_true", help="打印特征摘要、解读全文与引用片段")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"找不到测试图片：{image_path}")
        print("先生成一张合成测试人脸：python scripts/make_test_face.py")
        return 2

    print(f"\n== 1. 健康检查 {base}/api/health ==")
    status, health, _ = http("GET", f"{base}/api/health")
    check("健康检查返回 200", status == 200, f"status={status}")
    if isinstance(health, dict):
        check("status == ok", health.get("status") == "ok")
        engines = health.get("engines", {})
        check("至少有一个可用的人脸检测引擎",
              bool(engines.get("mediapipe") or engines.get("opencv_haar") or engines.get("cloud_configured")),
              f"engines={engines}")
        kb = health.get("kb", {})
        check("知识库已自动导入默认知识", kb.get("chunks", 0) > 0, f"chunks={kb.get('chunks')}")
    else:
        print("  健康检查返回内容异常：", health)
        return 1

    print("\n== 2. 游客账号 ==")
    status, account, _ = post_json(f"{base}/api/auth/guest", {})
    check("游客登录返回 200", status == 200, f"status={status}")
    token = account.get("access_token") if isinstance(account, dict) else None
    if not token:
        print("  未取得 token，后续用例无法继续")
        return 1
    check("拿到 JWT token", bool(token), f"user={account.get('user', {}).get('username')}")

    print("\n== 3. 未登录访问分析接口应被拒绝 ==")
    status, _, _ = post_file(f"{base}/api/face/analyze", image_path, token="")
    check("无 token 时返回 401", status == 401, f"status={status}")

    print(f"\n== 4. 面相分析（上传 {image_path.name}）==")
    status, result, _ = post_file(f"{base}/api/face/analyze", image_path, token=token)
    if status != 200:
        check("分析接口返回 200", False, f"status={status} detail={result if isinstance(result, str) else result}")
        _summary()
        return 1
    check("分析接口返回 200", True)
    record_id = result.get("record_id")
    features = result.get("features", {})
    reading = result.get("reading", "")
    kb_refs = result.get("kb_refs", [])
    check("返回结构化特征", isinstance(features, dict) and "three_courts" in features)
    courts = features.get("three_courts", {})
    parts = [courts.get(key, {}).get("ratio") or 0 for key in ("upper", "middle", "lower")]
    check("三庭是比例值且在合理区间", all(0.05 < value < 0.7 for value in parts),
          f"upper={parts[0]} middle={parts[1]} lower={parts[2]}")
    check("三庭之和约等于 1（归一化正确）", abs(sum(parts) - 1.0) < 0.02, f"sum={round(sum(parts), 4)}")
    check("识别出脸型", bool(features.get("face_shape", {}).get("label")),
          f"脸型={features.get('face_shape', {}).get('label')}")
    check("解读长度在 120 字以上", len(reading) >= 120, f"length={len(reading)}")
    check("解读带合规免责声明", "传统文化" in reading or "参考" in reading)
    check("RAG 检索到知识片段", len(kb_refs) > 0, f"hits={len(kb_refs)}")
    check("解读来源已标注", bool(result.get("llm_provider")), f"provider={result.get('llm_provider')}")
    check("照片地址带签名票据", "token=" in (result.get("image_url") or ""))

    if args.verbose:
        print("\n  --- 特征摘要 ---")
        for line in features.get("summary", []):
            print("   ", line)
        print(f"    检索词：{features.get('kb_query')}")
        print("\n  --- 解读全文 ---")
        for line in reading.splitlines():
            print("   ", line)
        print("\n  --- 引用片段 ---")
        for ref in kb_refs:
            print(f"    《{ref.get('doc_title')}》 score={ref.get('score')} :: {ref.get('excerpt', '')[:60]}…")

    print("\n== 5. 历史记录 ==")
    status, history, _ = http("GET", f"{base}/api/history", token=token)
    check("历史列表返回 200", status == 200 and isinstance(history, list), f"status={status}")
    check("历史里能看到刚才的记录", isinstance(history, list) and any(
        item.get("record_id") == record_id for item in history))

    status, detail, _ = http("GET", f"{base}/api/history/{record_id}", token=token)
    check("记录详情可读取", status == 200 and detail.get("record_id") == record_id)
    check("详情包含特征与解读", bool(detail.get("features")) and bool(detail.get("reading")))

    print("\n== 6. 知识库检索接口 ==")
    status, query_result, _ = post_json(f"{base}/api/kb/query", {"query": "三庭 五官 面相", "top_k": 3}, token=token)
    check("检索接口返回 200", status == 200, f"status={status}")
    check("检索有命中", bool(query_result.get("hits")) if isinstance(query_result, dict) else False,
          f"hits={len(query_result.get('hits', [])) if isinstance(query_result, dict) else 0}")

    print("\n== 7. 私有照片签名访问 ==")
    image_url = result.get("image_url") or ""
    status, blob, _ = http("GET", base + image_url)
    check("签名地址能取到图片", status == 200 and isinstance(blob, bytes) and len(blob) > 1000,
          f"status={status} bytes={len(blob) if isinstance(blob, bytes) else 'n/a'}")
    status, _, _ = http("GET", f"{base}/api/face/image/{record_id}?token=1.abc")
    check("伪造票据被拒绝", status == 403, f"status={status}")

    print("\n== 8. 删除与权限隔离 ==")
    if not args.keep:
        status, deleted, _ = http("DELETE", f"{base}/api/history/{record_id}", token=token)
        check("删除记录成功", status == 200 and deleted.get("deleted") is True, f"status={status}")
        status, _, _ = http("GET", f"{base}/api/history/{record_id}", token=token)
        check("删除后详情返回 404", status == 404, f"status={status}")

    status, other, _ = post_json(f"{base}/api/auth/guest", {})
    other_token = other.get("access_token") if isinstance(other, dict) else None
    if other_token:
        status, _, _ = http("GET", f"{base}/api/history/{record_id}", token=other_token)
        check("其他账号无法读取该记录", status == 404, f"status={status}")

    return _summary()


def _summary() -> int:
    print("\n================ 自测结果 ================")
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  - 失败：{name}")
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
