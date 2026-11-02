"""Bounded retrieval, validation, and one-shot query recovery workflow."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from stylemate.domain.models import SkillOutcome, SkillSpec
from stylemate.rag.models import RetrievalHit
from stylemate.rag.query_planning import knowledge_compatible, material_subjects
from stylemate.rag.question import QuestionContext
from stylemate.rag.support import CARE, SupportDecision, support_decision
from stylemate.skills.runtime import SkillRun


class KnowledgeQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=6)


class KnowledgeQASkill:
    spec = SkillSpec(
        name="knowledge_qa",
        description="混合检索已审核知识与用户资料，检查来源格式并在空召回时最多改写一次。",
        input_model=KnowledgeQuery,
        output_model=SkillOutcome,
        allowed_tools=("hybrid_retrieval", "source_validation", "query_rewrite"),
        max_steps=3,
        fallback_strategy="一次改写后仍无带来源结果时明确返回未找到答案。",
    )

    def __init__(self, retriever):
        self.retriever = retriever

    def run(
        self,
        owner_id: str,
        conversation_id: str,
        request: KnowledgeQuery,
        *,
        question: QuestionContext | None = None,
    ) -> SkillOutcome:
        run = SkillRun(self.spec, request, trace_name="KnowledgeQASkill")
        request = run.request
        if question and (question.owner_id, question.conversation_id) != (owner_id, conversation_id):
            raise ValueError("question_identity_mismatch")
        original = question.constraint_query if question else request.query
        care = bool(CARE.search(original) or CARE.search(request.query))
        # A compact material argument must not erase the requested operation.
        # Trusted text already receives bounded lexical expansion in retrieval.
        query = _normalize_query(original if question else request.query)
        with run.step("hybrid_retrieval") as step:
            hits = self.retriever.search(query, owner_id, conversation_id, top_k=request.top_k) if self.retriever else []
            if not hits:
                step.status = "fallback"
        if self.retriever is None:
            return self._outcome(original, query, [], run, attempts=0, rewritten=False, care=care, question=question)
        with run.step("source_validation") as step:
            valid_hits = _valid_cited_hits(hits, original)
            if not valid_hits:
                step.status = "fallback"
        attempts = 1
        rewritten = False
        effective_query = query
        if not valid_hits:
            rewritten = True
            attempts = 2
            with run.step("query_rewrite") as step:
                effective_query = _rewrite_query(query)
                retry_hits = self.retriever.search(effective_query, owner_id, conversation_id, top_k=request.top_k)
                valid_hits = _valid_cited_hits(retry_hits, original)
                if not valid_hits:
                    step.status = "fallback"
        return self._outcome(original, effective_query, valid_hits, run,
                             attempts=attempts, rewritten=rewritten, care=care, question=question)

    def _outcome(
        self,
        original_query: str,
        effective_query: str,
        hits: list[RetrievalHit],
        run: SkillRun,
        *,
        attempts: int,
        rewritten: bool,
        care: bool = False,
        question: QuestionContext | None = None,
    ) -> SkillOutcome:
        decision = support_decision(original_query, hits, care=care)
        if question and question.recorded_material:
            declared = material_subjects(question.original_query)
            recorded = material_subjects(question.recorded_material)
            if declared and recorded and declared - recorded:
                decision = SupportDecision("conflict", missing=("garment_material_confirmation",), message=
                    "本轮提到的材质与所选衣物记录不一致，请先核实当前这件的成分和洗护标签；不会擅自选择另一种材料的操作。")
        hits = list(decision.hits)
        results = [_result_payload(hit) for hit in hits]
        status = "success" if results else "fallback"
        return run.finish(
            status=status,
            data={
                "query": original_query,
                "effective_query": effective_query,
                "query_rewritten": rewritten,
                "attempts": attempts,
                "validation_scope": "source_identity_material_product_action_rules_not_general_entailment",
                "answer_status": ("evidence_found" if results else "needs_clarification" if decision.missing else "not_found"),
                "support_status": decision.status,
                "missing_conditions": list(decision.missing),
                "care_boundary": care,
                "message": decision.message,
                "evidence_ids": [hit.chunk_id or hit.record_id or hit.document_id for hit in hits],
                "information_provenance": ("none" if not hits else "user_supplied_document" if any(h.document_id for h in hits) else "reviewed_builtin_summary"),
                "results": results,
                "sources": [
                    {
                        "title": item["title"],
                        "url": item["source_url"],
                        "source_name": item["source_name"],
                    }
                    for item in results
                ],
            },
            message=decision.message or ("已找到限定范围内的来源资料。" if results else "未找到适用证据。"),
        )


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def _rewrite_query(query: str) -> str:
    aliases = {
        "冲锋衣": "功能外套",
        "防泼水": "耐久拒水",
        "尺码号": "品牌尺码表",
    }
    rewritten = query
    for source, target in aliases.items():
        rewritten = rewritten.replace(source, target)
    rewritten = re.sub(r"[吗呢么？?]+$", "", rewritten).strip()
    if rewritten == query:
        # Do not introduce a new task (e.g. care) that can override the user's
        # original size/color intent or make unrelated records appear relevant.
        rewritten = f"{rewritten} 相关资料"
    return rewritten[:4000]


def _valid_cited_hits(hits: list[RetrievalHit], original_query: str) -> list[RetrievalHit]:
    return [
        hit
        for hit in hits
        if hit.snippet.strip()
        and hit.source_url.startswith(("https://", "http://", "user-document://"))
        and (_document_material_compatible(hit, original_query) if hit.document_id or hit.source_url.startswith("user-document://") else knowledge_compatible(
            original_query, title=hit.title, content=hit.snippet, topic=hit.topic,
            record_id=hit.record_id,
        ))
    ]


def _document_material_compatible(hit: RetrievalHit, query: str) -> bool:
    requested = material_subjects(query)
    available = material_subjects(hit.title + " " + hit.snippet)
    # A product label may omit fibre information. An explicit different fibre
    # is still a mismatch even when the document belongs to this conversation.
    return not (requested and available and not requested.intersection(available))


def _result_payload(hit: RetrievalHit) -> dict:
    return {
        "title": hit.title,
        "snippet": hit.snippet[:400],
        "source_name": hit.source_name,
        "source_url": str(hit.source_url),
        "topic": hit.topic,
        "score": hit.score,
        "record_id": hit.record_id,
        "document_id": hit.document_id,
        "chunk_id": hit.chunk_id,
        "page_number": hit.page_number,
        "section_title": hit.section_title,
        "source_locator": hit.source_locator,
        "review_status": hit.review_status,
        "applicability": hit.applicability,
        "limitations": hit.limitations,
    }
