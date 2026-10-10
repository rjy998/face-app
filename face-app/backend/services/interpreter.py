"""
面相解读（第二段 AI）：把「人脸特征 + 知识库片段」喂给大模型生成解读。

三级策略，保证任何环境都有结果：
  1. 豆包（火山方舟）API —— 默认方案，.env 填 ARK_API_KEY 即启用；
  2. OpenAI 兼容接口 —— 换 base_url 即可用 DeepSeek / Kimi / 通义 / 自建 LLM；
  3. 本地模板拼装 —— 没有密钥或调用失败时的兜底，内容仍然来自「特征 + 知识库片段」，
     不编造、不承诺，保证 /api/face/analyze 永远返回可用结果。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from config import settings
from services import compliance
from services.rag import RetrievedChunk

logger = logging.getLogger("face_app.interpreter")

SYSTEM_PROMPT = """你是一位面相学文化顾问，向用户做「传统面相文化解读」。

必须遵守：
1. 只依据【人脸特征】与【参考资料】作答，不得编造参考资料里没有的说法；
2. **解读主体必须是面相学内容**：把测到的特征对应到传统相学的说法与术语，例如
   三庭/上停·中停·下停与年龄段的对应、眉为保寿官、眼为监察官、鼻为审辨官、
   口唇为出纳官、印堂与气色、脸型与整体气质等；参考资料里的说法要直接用上；
3. **吉凶两面都要讲**（重要）：既要指出**偏吉、可取之处**，也要指出**相理上不足、
   欠佳、需要留意之处**。逐部位对照着说，该说不足就说不足，不要只拣好话，
   也不要为了委婉把「欠佳」淡化成「不错」；传统上认为偏弱、偏薄、偏窄、
   不够开阔、需多留意的地方，要如实讲出来，并说明传统上认为这提示什么；
4. **不要写与面相无关的通用健康/心理/作息建议**（如「规律睡眠」「练习先听后讲」
   「注意双侧咀嚼」这类，除非参考资料里确实有对应说法）；
5. **只对【人脸特征】里实际测量到的部位下判断**：
   - 标记为 `"measured": false` 或「未测得」的项目，不要对其高低、宽窄、吉凶下任何判断，
     也不要在解读里出现（例如三庭里上庭未测得，就只谈中庭与下庭）；
   - 照片测不到的部位（耳朵形态、痣、法令纹、牙齿等）同样不要提及；
   - 额部宽窄看 `forehead.width_label`（额宽/颧宽），额部高低才看上庭比例，两者不要混用；
6. **不足之处一律写成建议，不要写成贬义评价**（重要）：例如不要写「鼻翼过厚易显优柔」
   这种定性判断，而写成「在理财与决断上宜多留意」；不要写「格局稍欠舒朗」，
   写成「可多养沉静专注之神态」。好坏都要覆盖，但「不好」的部分以**怎么做**的形式给出；
7. 只输出两段：
   【结论】4-6 句：总述整体格局与相理上可取之处，并逐部位给出**中性或偏吉**的描述
   （可以用「偏吉」「较佳」「舒展」「端正」这类相学评价语）；
   【注意事项】3-4 条：把上面那些需要留意的地方，逐条改写成建议
   （「宜…」「可多留意…」「建议…」），仍是相学角度（宜养什么神态、什么样的习惯）；
8. 语气客观、不讨好也不吓唬；
9. 严禁：生死的预测、疾病诊断、绝对化承诺（必定/一定/注定）、恐吓性表述（大凶/克夫/寿元/破财等），
   也不要复述参考资料里可能残留的具体病名；
10. 全文 300-400 字，用【】分段，不要 Markdown 标题符号。"""

USER_PROMPT_TEMPLATE = """【人脸特征】
{features_json}

【参考资料（来自面相知识库，已做中性化处理，请直接引用其中的说法与术语）】
{retrieved_chunks}

【请生成解读：只给【结论】和【注意事项】两段；不足的部分写成建议】"""


# ---------------------------------------------------------------------------
# Prompt 组装
# ---------------------------------------------------------------------------
def _chunk_heading(text: str) -> str:
    """取出切片前缀里的标题路径（形如「【五官 > 耳 > 耳型分类】…」）。"""
    stripped = text.strip()
    if stripped.startswith("【"):
        end = stripped.find("】")
        if end > 0:
            return stripped[1:end]
    return "正文"


def _chunk_body(text: str) -> str:
    """取切片正文（去掉标题前缀）。"""
    stripped = text.strip()
    if stripped.startswith("【"):
        end = stripped.find("】")
        if end > 0:
            return stripped[end + 1 :].strip()
    return stripped


def build_messages(features: dict[str, Any], refs: list[RetrievedChunk]) -> list[dict[str, str]]:
    """把特征与检索片段组装成对话消息（RAG 的注入点）。"""
    slim = {
        "脸型": features.get("face_shape", {}).get("label"),
        "三庭": features.get("three_courts", {}).get("judgement"),
        "五眼": features.get("five_eyes", {}).get("judgement"),
        "五官": {
            key: value.get("label")
            for key, value in (features.get("facial_features") or {}).items()
            if isinstance(value, dict)
        },
        "气色": features.get("complexion", {}).get("label"),
        "对称度": features.get("symmetry", {}).get("label"),
        "备注": features.get("warnings") or [],
    }
    chunks_text = "\n\n".join(
        f"{index + 1}. 出处《{ref.doc_title}》｜{_chunk_heading(ref.text)}\n{_chunk_body(ref.text)[:600]}"
        for index, ref in enumerate(refs)
    ) or "（知识库暂无相关内容，请仅依据人脸特征做温和的传统文化解读）"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                features_json=json.dumps(slim, ensure_ascii=False, indent=2),
                retrieved_chunks=chunks_text,
            ),
        },
    ]


# ---------------------------------------------------------------------------
# 大模型调用
#   协议一：OpenAI 兼容（/chat/completions）—— 豆包方舟、DeepSeek、Kimi、通义、自建 vLLM
#   协议二：Anthropic 兼容（/v1/messages）—— Claude、MiniMax（https://api.minimax.cn/anthropic）
# ---------------------------------------------------------------------------
def _chat_completion_openai(
    base_url: str, api_key: str, model: str, messages: list[dict[str, str]],
    max_tokens: int | None = None, temperature: float = 0.7,
) -> str:
    import httpx  # 延迟导入，避免无网络依赖时启动失败

    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
        },
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def _chat_completion_anthropic(
    base_url: str, api_key: str, model: str, messages: list[dict[str, str]],
    max_tokens: int | None = None, temperature: float = 0.7,
) -> str:
    """
    Anthropic Messages API（MiniMax 的 Anthropic 兼容面即走这里）。

    与 OpenAI 协议的两点差异：
      1. system 提示是顶层参数，不放在 messages 里；
      2. 返回体是 content 块数组，需要跳过 thinking（推理）块，只取 text 块。
    """
    import httpx

    system_text = "\n".join(item["content"] for item in messages if item.get("role") == "system")
    chat_messages = [item for item in messages if item.get("role") != "system"]

    response = httpx.post(
        f"{base_url.rstrip('/')}/v1/messages",
        headers={
            "Authorization": f"Bearer {api_key}",   # MiniMax 用 Bearer
            "x-api-key": api_key,                   # Claude 官方用 x-api-key，两个都带上更通用
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "system": system_text,
            "messages": chat_messages,
            "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
            "temperature": temperature,
        },
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    blocks = payload.get("content") or []
    texts = [
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    if not texts:  # 个别实现只返回字符串
        texts = [payload.get("completion") or payload.get("text") or ""]
    return "".join(texts).strip()


def _chat_completion(
    provider: dict[str, str], messages: list[dict[str, str]],
    max_tokens: int | None = None, temperature: float = 0.7,
) -> str:
    """按协议分发到对应实现。"""
    if provider["protocol"] == "anthropic":
        return _chat_completion_anthropic(
            provider["base_url"], provider["api_key"], provider["model"], messages,
            max_tokens=max_tokens, temperature=temperature,
        )
    return _chat_completion_openai(
        provider["base_url"], provider["api_key"], provider["model"], messages,
        max_tokens=max_tokens, temperature=temperature,
    )


def _resolve_provider() -> dict[str, str] | None:
    """
    决定用哪个大模型：优先用 LLM_PROVIDER 指定的，缺密钥时按 anthropic → openai → doubao 顺序兜。
    返回 {"name", "protocol", "base_url", "api_key", "model"}；都没配置则 None（走本地模板）。
    """
    provider = (settings.LLM_PROVIDER or "template").lower()
    catalog = {
        "doubao": ("openai", settings.ARK_BASE_URL, settings.ARK_API_KEY, settings.ARK_MODEL),
        "openai": ("openai", settings.OPENAI_BASE_URL, settings.OPENAI_API_KEY, settings.OPENAI_MODEL),
        "anthropic": ("anthropic", settings.ANTHROPIC_BASE_URL, settings.ANTHROPIC_API_KEY, settings.ANTHROPIC_MODEL),
        "minimax": ("anthropic", settings.ANTHROPIC_BASE_URL, settings.ANTHROPIC_API_KEY, settings.ANTHROPIC_MODEL),
    }
    order = [provider] + [name for name in ("anthropic", "openai", "doubao") if name != provider]
    for name in order:
        entry = catalog.get(name)
        if entry and entry[2]:
            protocol, base_url, api_key, model = entry
            return {
                "name": name,
                "protocol": protocol,
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
            }
    return None


# ---------------------------------------------------------------------------
# 通用补全：给知识库整理等内部任务用（失败时抛异常，不静默兜底）
# ---------------------------------------------------------------------------
class LLMNotConfigured(RuntimeError):
    """没有配置任何大模型密钥。"""


def complete(
    system_prompt: str, user_prompt: str, max_tokens: int | None = None, temperature: float = 0.3
) -> tuple[str, str]:
    """
    通用一次性补全，返回 (文本, provider 名称)。

    与 generate_reading 的区别：这里**不做模板兜底**——知识库整理这类任务，
    安静地降级成模板会产出无意义的内容，所以直接抛异常让调用方处理。
    """
    provider = _resolve_provider()
    if provider is None:
        raise LLMNotConfigured(
            "未配置大模型密钥：请在 backend/.env 里填写 ANTHROPIC_API_KEY / ARK_API_KEY / OPENAI_API_KEY 之一"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = _chat_completion(provider, messages, max_tokens=max_tokens, temperature=temperature)
    return text, provider["name"]


# ---------------------------------------------------------------------------
# 兜底模板
# ---------------------------------------------------------------------------
def template_reading(features: dict[str, Any], refs: list[RetrievedChunk]) -> str:
    """本地拼装解读：同样只使用「特征 + 知识片段」，不编造。"""
    shape = features.get("face_shape", {})
    courts = features.get("three_courts", {})
    eyes = features.get("five_eyes", {})
    facial = features.get("facial_features", {})
    complexion = features.get("complexion", {})
    symmetry = features.get("symmetry", {})

    impression = (
        f"从照片看，你的面部轮廓偏向「{shape.get('label', '匀称')}」"
        f"（{shape.get('reason', '长宽比例适中')}），{symmetry.get('label', '左右对称度良好')}，"
        f"整体给人温和、容易亲近的印象。"
    )

    parts = [
        f"三庭方面，{courts.get('judgement', '三庭比例较为均衡')}；"
        f"传统说法中，比例协调的人常给人稳重、做事有条理的印象。",
        f"五官方面，{eyes.get('judgement', '整体接近五眼比例')}；"
        f"{facial.get('brow', {}).get('label', '眉眼距离适中')}，"
        f"{facial.get('eye', {}).get('label', '眼型适中')}，"
        f"{facial.get('nose', {}).get('label', '鼻宽适中')}，"
        f"{facial.get('mouth', {}).get('label', '唇厚适中')}。",
        f"气色方面，{complexion.get('label', '肤色观感自然')}；"
        f"{complexion.get('caveat', '气色受光线影响，仅供参考')}。",
    ]

    hints = _reference_hints(refs)
    if hints:
        parts.append(f"以上判断参考了知识库中{hints}，仅作传统文化角度的对照。")

    tips = (
        "面相是传统文化中的一种趣味观察方式，反映的是长期生活习惯与精神状态的影子，"
        "并不决定任何结果。保持规律作息与舒展的眉眼神态，比任何「面相」都更耐看。"
    )
    return "\n".join([f"【整体印象】{impression}", f"【分项解读】{' '.join(parts)}", f"【温馨提示】{tips}"])


def _reference_hints(refs: list[RetrievedChunk]) -> str:
    """把命中片段整理成「《文档》中关于『小节』的记述」这种可读的引用说明（去重、最多两条）。"""
    hints: list[str] = []
    for ref in refs[:3]:
        label = f"《{ref.doc_title}》中关于「{_keyword(ref.text)}」的记述"
        if label not in hints:
            hints.append(label)
    return "、".join(hints[:2])


def _keyword(text: str) -> str:
    """
    从知识片段里取一个短关键词，用于模板文案的引用说明。

    片段格式为「【一级标题 > 二级标题】正文…」，优先用最细一级的标题。
    """
    body = text.strip()
    heading = ""
    if body.startswith("【"):
        end = body.find("】")
        if end > 0:
            heading = body[1:end]
            body = body[end + 1 :]
    if heading:
        last = heading.split(">")[-1].strip()
        if last:
            return last[:16]
    for segment in re.split(r"[，。；：、\s]", body):
        if len(segment.strip()) >= 3:
            return segment.strip()[:16]
    return "面相常识"


# ---------------------------------------------------------------------------
# 对外主函数
# ---------------------------------------------------------------------------
def generate_reading(features: dict[str, Any], refs: list[RetrievedChunk]) -> tuple[str, str]:
    """
    生成解读，返回 (解读文本, 来源标识)。

    来源标识取值：doubao / openai / template / template(fallback:<原因>)
    """
    resolved = _resolve_provider()
    if resolved is not None:
        name = resolved["name"]
        try:
            messages = build_messages(features, refs)
            text = _chat_completion(resolved, messages)
            if text:
                return compliance.ensure_disclaimer(compliance.sanitize(text)), name
            logger.warning("大模型返回空内容，改用模板兜底")
            return compliance.ensure_disclaimer(compliance.sanitize(template_reading(features, refs))), "template(empty)"
        except Exception as exc:  # noqa: BLE001
            logger.warning("调用 %s 失败（%s），改用模板兜底", name, exc)
            return (
                compliance.ensure_disclaimer(compliance.sanitize(template_reading(features, refs))),
                f"template(fallback:{type(exc).__name__})",
            )

    logger.info("未配置大模型密钥，使用本地模板生成解读")
    return compliance.ensure_disclaimer(compliance.sanitize(template_reading(features, refs))), "template"
