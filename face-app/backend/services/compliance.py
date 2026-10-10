"""
合规过滤：解读文案不得出现迷信、绝对化、恐吓性表述。

两道防线：
1. 生成前——Prompt 里写死规则（见 interpreter.py）；
2. 生成后——本模块做关键词替换 + 免责声明强制追加，确保线上文案可控。
"""

from __future__ import annotations

import re

from config import settings

# 命中后的中性替换：不删除整句，避免语义断裂
_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("命中注定", "传统说法中常被这样形容"),
    ("改运", "调整心态"),
    ("转运", "换个心情"),
    ("消灾", "安心"),
    ("辟邪", "安心"),
    ("法事", "传统文化活动"),
    ("克夫", "传统说法认为需要多沟通"),
    ("克妻", "传统说法认为需要多沟通"),
    ("克父母", "传统说法认为需多陪伴家人"),
    ("大凶", "需要多留意"),
    ("凶兆", "值得留意之处"),
    ("必发财", "有机会获得发展"),
    ("一定发财", "有机会获得发展"),
    ("必定富贵", "有机会稳步提升"),
    ("保证发财", "有机会稳步提升"),
    ("必然成功", "更有可能顺利"),
    ("短命", "要多注意休息"),
    ("绝后", "家庭规划因人而异"),
    # 古籍整理时常见的命定 / 恐吓表述
    ("主贫", "传统相学认为生活较为朴素"),
    ("主病", "传统相学认为需留意日常作息"),
    ("主贱", "传统相学中有此一形容"),
    ("贫贱", "生活较为朴素"),
    ("贱格", "传统说法中的一种形容"),
    ("寿元较短", "传统相学认为需注意保养"),
    ("寿元不长", "传统相学认为需注意保养"),
    ("寿元", "传统相学认为需注意保养"),
    ("寿数", "年岁"),
    ("寿终正寝", "安享晚年"),
    ("父母双亡", "父母的身体健康值得多留意"),
    ("较早离世", "健康方面需多留意"),
    ("离世", "健康方面需多留意"),
    ("易逝", "健康值得多留意"),
    ("早逝", "健康欠安"),
    ("早亡", "健康欠安"),
    ("先损", "的健康值得多留意"),
    ("先亡", "的健康值得多留意"),
    ("损父", "父亲的健康值得多留意"),
    ("损母", "母亲的健康值得多留意"),
    ("短寿", "需注意保养"),
    ("夭折", "需注意保养"),
    ("折寿", "需注意保养"),
    ("孤寡", "性格较为独立"),
    ("刑克", "相处中宜多沟通"),
    ("无子", "家庭规划因人而异"),
    ("破财", "财务上宜谨慎"),
    ("牢狱", "行事宜守法谨慎"),
    ("血光", "需注意安全"),
    ("横死", "需注意安全"),
    ("苦命", "早年较为辛苦"),
    ("残疾", "身体状况因人而异"),
    ("淫贱", "传统说法中的一种形容"),
    # 生育 / 胎儿性别预测：长词在前，避免被短词先替换（如「生男孩」被「生男」截断）
    ("生男孩", "传统相学中有相关观察角度"),
    ("生女孩", "传统相学中有相关观察角度"),
    ("怀男孩", "传统相学中有相关观察角度"),
    ("怀女孩", "传统相学中有相关观察角度"),
    ("头胎", "传统相学中有相关观察角度"),
    ("生男", "传统相学中有相关观察角度"),
    ("生女", "传统相学中有相关观察角度"),
)

_EXTRA_BANNED = ("包治", "死", "算命", "看相定命")


def find_violations(text: str) -> list[str]:
    """返回命中的违禁词列表（用于日志与自测）。"""
    hits = [word for word in settings.BANNED_PHRASES if word in text]
    hits += [word for word in _EXTRA_BANNED if word in text]
    return sorted(set(hits))


def sanitize(text: str) -> str:
    """替换绝对化 / 迷信表述，并去掉可能的绝对断言句式。"""
    if not text:
        return ""
    result = text
    for bad, good in _REPLACEMENTS:
        result = result.replace(bad, good)
    # 兜底：仍未被替换表覆盖的违禁词，直接软化
    for bad in settings.BANNED_PHRASES:
        if bad in result:
            result = result.replace(bad, "（传统说法）")
    # 「一定会 / 必定会 / 100%」这类断言统一软化
    result = re.sub(r"(一定|必定|必然|绝对|100%|百分之百)会", "有机会", result)
    result = re.sub(r"(一定|必定|必然|绝对)能", "有机会", result)
    # 折叠替换造成的相邻重复（如「克夫克妻」→「…多沟通」连写两遍），避免生成赘文
    for _bad, good in _REPLACEMENTS:
        result = re.sub(f"(?:{re.escape(good)})+", good, result)
    return result


def ensure_disclaimer(text: str) -> str:
    """确保结尾带免责声明（可用 COMPLIANCE_DISCLAIMER_ENABLED=false 关闭；重复出现时不重复追加）。"""
    if not getattr(settings, "COMPLIANCE_DISCLAIMER_ENABLED", True):
        return text.rstrip()
    disclaimer = settings.COMPLIANCE_DISCLAIMER
    if disclaimer in text:
        return text
    return f"{text.rstrip()}\n\n—— {disclaimer}"
