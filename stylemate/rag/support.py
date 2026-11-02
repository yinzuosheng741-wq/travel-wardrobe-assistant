"""Conservative evidence-use rules, not a general semantic entailment model."""

import re
from dataclasses import dataclass

from stylemate.rag.models import RetrievalHit
from stylemate.rag.query_planning import is_outside_builtin_scope, material_subjects, requests_exact_product_measurement

CARE = re.compile(r"洗|清洁|护理|保养|烘|漂白|熨|去渍|收纳|(?<![A-Za-z])(?:wash|care)(?![A-Za-z])", re.I)
_PERMISSION = re.compile(r"(?:能|可|直接|允许|适合|是否).{0,15}(?:机洗|洗衣机|水洗|手洗|烘干|熨烫|漂白)")
_CHEMICAL = re.compile(r"化学剂|溶剂|配方|浓度|(?:血渍|咖啡渍|油渍).{0,20}(?:用什么|哪种|哪一种|去污)")
_LABEL_NO = re.compile(r"(?:洗标|标签).{0,12}(?:写着?|注明|标注|显示|要求|明确).{0,8}(?:不可|禁止|不能)(烘干|机洗|水洗|漂白|熨烫)")


@dataclass(frozen=True)
class SupportDecision:
    status: str
    hits: tuple[RetrievalHit, ...] = ()
    missing: tuple[str, ...] = ()
    message: str = ""


def exact_measurement_hits(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """An uploaded document must contain matching product and numeric evidence."""
    products = re.findall(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9-]*\d[A-Za-z0-9-]*", query)
    sizes = re.findall(r"(?<![A-Za-z])(?:XXS|XS|S|M|L|XL|XXL|\dXL)码", query, re.I)
    dimensions = [word for word in ("胸围", "腰围", "肩宽", "衣长", "袖长", "裤长", "胸宽") if word in query]
    selected = []
    for hit in hits:
        if not hit.document_id or not products or not dimensions:
            continue
        text = hit.title + " " + hit.snippet
        if not all(token.casefold() in text.casefold() for token in products + sizes):
            continue
        if any(re.search(rf"{word}[^。\n]{{0,16}}\d+(?:\.\d+)?\s*(?:厘米|公分|cm)(?![A-Za-z])", text, re.I)
               for word in dimensions):
            selected.append(hit)
    return selected


def support_decision(query: str, hits: list[RetrievalHit], *, care: bool = False) -> SupportDecision:
    care = care or bool(CARE.search(query))
    if is_outside_builtin_scope(query) and not any(hit.document_id for hit in hits):
        return SupportDecision("out_of_scope", message="当前知识库聚焦衣物与出行准备，没有适用于这个问题的证据。")
    if requests_exact_product_measurement(query):
        exact = exact_measurement_hits(query, hits)
        if not exact:
            return SupportDecision("needs_clarification", missing=("product_size_chart",), message=
                "当前资料没有这款产品的具体尺寸数据。请提供准确品牌、款号及对应尺码表或测量数据；通用量衣指南不能代替该款数据。")
        return SupportDecision("supported", tuple(exact))
    if care:
        prohibition = _LABEL_NO.search(query)
        if prohibition:
            action = prohibition[1]
            return SupportDecision("conflict", missing=("product_care_label_confirmation",), message=
                f"你提供的洗标说明是不可{action}，请遵守这一限制；降低温度或缩短时间不构成{action}许可。若洗标含义有疑问，请向厂家核实。")
        if re.search(r"(?:洗标|标签).{0,15}(?:仅|只能|只可|只允许).{0,5}(?:专业)?干洗", query):
            return SupportDecision("conflict", missing=("product_care_label_confirmation",), message=
                "你提供的洗标要求只能专业干洗，请遵循该件标签；通用羊毛资料不能据此授权手洗或机洗。可向厂家核实专业护理要求。")
        materials = material_subjects(query) - {"goretex"}
        if _CHEMICAL.search(query):
            return SupportDecision("partial", missing=("specific_treatment_evidence",), message=
                "当前资料不足以支持你要求的具体去污剂、溶剂或配方。一般材质护理不等于该污渍处理许可，请提供整件洗标及厂家针对性说明。")
        compound = bool(re.search(r"混纺|拼接|饰片|装饰|涂层|衬里", query))
        operational = bool(_PERMISSION.search(query) or re.search(r"怎么洗|如何清洗|如何洗|怎样洗|怎么清洗", query))
        if (operational and compound) or (len(materials) > 1 and not re.search(r"区别|比较|分别|不同", query)):
            return SupportDecision("partial", missing=("whole_garment_care_label",), message=
                "仅凭一种纤维的通用建议，无法确认混纺、衬里或装饰所在整件衣物的处理许可。请提供整件洗标和厂家说明后再判断。")
        if "shakedry" in query.casefold():
            scoped = []
            for hit in hits:
                clauses = [part for part in re.split(r"(?<=[。！？])", hit.snippet) if "shakedry" in part.casefold()]
                if clauses:
                    scoped.append(hit.model_copy(update={"snippet": "".join(clauses),
                        "applicability": "仅引用来源中关于 SHAKEDRY 的排除条款；不授予普通外套处理许可。"}))
            hits = scoped
            hits.sort(key=lambda hit: sum(word in hit.snippet for word in ("烘干", "熨烫", "DWR")), reverse=True)
        elif _PERMISSION.search(query) and not any(hit.document_id for hit in hits):
            # Return only evidence about the decision boundary, never an
            # unrelated preparation step as a yes/no treatment permission.
            boundary = [hit for hit in hits if hit.record_id in {"fabric-wool-label", "care-label-complete", "care-symbol-sequence"}]
            if boundary:
                return SupportDecision("partial", tuple(boundary), ("product_care_label",),
                    "这些资料只能说明应如何核对护理条件，尚不能确认你这件衣物可以采用该处理方式；请提供整件洗标。")
            return SupportDecision("needs_clarification", missing=("product_care_label",), message=
                "当前通用资料不能确认这件衣物的具体处理许可，请提供整件洗标或厂家护理说明。")
        if hits and not any(hit.document_id for hit in hits):
            actions = [pattern for pattern in (r"收纳|储存", r"干燥|晾晒|晾干|平放|变形", r"衣刷|刷|浮灰|浮尘", r"柔顺剂")
                       if re.search(pattern, query)]
            if actions:
                relevant = [hit for hit in hits if any(re.search(pattern, hit.snippet) for pattern in actions)]
                if relevant:
                    relevant.sort(key=lambda hit: sum(bool(re.search(pattern, hit.snippet)) for pattern in actions), reverse=True)
                    if re.search(r"收纳|存放|储存", query):
                        relevant.sort(key=lambda hit: hit.record_id == "storage-dry-before-fold", reverse=True)
                    hits = relevant
                else:
                    hits = []
        if len(materials) > 1 and hits:
            covered = set().union(*(material_subjects(hit.title + " " + hit.snippet) for hit in hits))
            if not materials <= covered:
                return SupportDecision("partial", tuple(hits), ("material_comparison_evidence",),
                    "当前只找到部分材质的资料，不能完成全部材质的对照，也不能相互套用处理方法；请补充缺少材料的厂家说明。")
    if not hits:
        return SupportDecision("not_found", message=(
            "当前没有足够适用的洗护资料；请提供这件衣物的成分、洗标或厂家护理说明。" if care else
            "知识库暂未找到适用于当前问题的有效证据。"))
    return SupportDecision("supported", tuple(hits))


def evidence_text(data: dict) -> str:
    """A bounded factual rendering shared by the UI and guarded model graph."""
    lines = [data.get("message", "")] if data.get("support_status") != "supported" else []
    results = data.get("results", [])
    if results:
        first = results[0]
        if first.get("document_id"):
            lines.append("根据你上传的资料：")
        chosen = ([r for r in results if r.get("document_id") == first.get("document_id")][:2]
                  if first.get("document_id") else results[:2] if data.get("care_boundary") else [first])
        for item in chosen:
            lines.append(item.get("snippet", ""))
            for title, key in (("适用范围", "applicability"), ("限制", "limitations")):
                if item.get(key):
                    lines.append(f"{title}：{item[key]}")
            lines.append(f"来源：{item.get('source_url', '')}")
    return "\n".join(line for line in lines if line) or "当前缺少适用于该问题的证据。"
