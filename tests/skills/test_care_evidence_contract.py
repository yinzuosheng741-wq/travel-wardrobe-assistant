import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import Barrier
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from evaluation.benchmark import harness
from stylemate.agent.execution import BoundedToolRunner
from stylemate.agent.graph import build_agent_graph
from stylemate.agent.routing import route_skills
from stylemate.agent.state import initial_state
from stylemate.rag.models import RetrievalHit
from stylemate.rag.query_planning import knowledge_compatible
from stylemate.rag.question import QuestionContext, current_question, question_scope
from stylemate.rag.support import support_decision
from stylemate.skills.knowledge_qa import KnowledgeQASkill, KnowledgeQuery
from tests.skills.test_knowledge_qa import SequenceRetriever


def test_original_numeric_requirement_survives_model_rewrite_and_graph_refuses_fabricated_final():
    class Model:
        calls = 0

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "rag_search", "args": {"query": "品牌尺码表测量方法"}, "id": "rewrite",
                }])
            raise AssertionError("missing evidence must not reach a fabricating final response")

    app = harness()
    model = Model()
    tools = app._toolkit("benchmark-owner", "turn")
    state = build_agent_graph(model, tools.tools).invoke(initial_state(
        "请查K72西装M码胸围多少厘米", owner_id="benchmark-owner", conversation_id="turn",
    ))
    assert model.calls == 1
    assert "尺码表" in state["messages"][-1].content
    assert state["sources"] == []


def test_question_context_is_frozen_and_restored_after_exception():
    question = QuestionContext("owner", "thread", "original")
    with pytest.raises(FrozenInstanceError):
        question.original_query = "model overwrite"
    with pytest.raises(RuntimeError), question_scope(question):
        assert current_question("owner", "thread") is question
        assert current_question("other", "thread") is None
        raise RuntimeError("injected")
    assert current_question("owner", "thread") is None


def test_parallel_tool_workers_keep_original_questions_separate_and_cleanup_context():
    runner = BoundedToolRunner(max_workers=2)
    barrier = Barrier(2)

    def request(owner, text):
        with question_scope(QuestionContext(owner, "turn", text)):
            def worker():
                barrier.wait(timeout=3)
                return current_question(owner, "turn").original_query, current_question("other-" + owner, "turn")
            return runner.submit(worker).result(timeout=4)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(request, owner, text) for owner, text in (("a", "cotton"), ("b", "acetate"))]
            assert [f.result(timeout=5) for f in futures] == [("cotton", None), ("acetate", None)]
        assert runner.submit(lambda: current_question("a", "turn")).result(timeout=2) is None
    finally:
        runner.shutdown()


@pytest.mark.parametrize("query", ["洗标写着不可烘干，可低温烘干吗", "羊毛洗标注明只能专业干洗，能手洗吗"])
def test_explicit_label_restrictions_cannot_be_removed_from_tool_arguments(query):
    app = harness()
    toolkit = app._toolkit("benchmark-owner", "turn")
    with question_scope(QuestionContext("benchmark-owner", "turn", query)):
        tool = next(t for t in toolkit.tools if t.name == "care_guide")
        payload = json.loads(tool.invoke({"material": "羊毛"}))
    assert payload["result"]["support_status"] == "conflict"
    assert payload["result"]["results"] == payload["sources"] == []
    assert payload["outcome"] == "needs_clarification"


def test_uploaded_generic_measurement_document_is_not_product_measurement_evidence():
    hit = RetrievalHit(title="K72 M码量衣指南", snippet="K72 M码胸围测量是沿着衣物测量，本指南没有该款的数值。",
                       source_name="User upload", source_url="user-document://generic", topic="wardrobe",
                       score=1, document_id="generic")
    result = KnowledgeQASkill(SequenceRetriever([[hit]])).run(
        "owner", "turn", KnowledgeQuery(query="K72 M码胸围多少厘米"),
    )
    assert result.data["support_status"] == "needs_clarification"
    assert result.data["missing_conditions"] == ["product_size_chart"]
    assert result.data["sources"] == []


def test_uploaded_matching_product_measurement_can_supply_user_attributed_evidence():
    hit = RetrievalHit(title="K72 M码实测", snippet="K72 M码胸围为104厘米。",
                       source_name="User upload", source_url="user-document://chart", topic="wardrobe",
                       score=1, document_id="chart")
    result = KnowledgeQASkill(SequenceRetriever([[hit]])).run(
        "owner", "turn", KnowledgeQuery(query="K72 M码胸围多少厘米"),
    )
    assert result.data["support_status"] == "supported"
    assert result.data["information_provenance"] == "user_supplied_document"
    assert result.data["results"][0]["snippet"] == hit.snippet


def test_question_identity_mismatch_is_rejected_before_searching():
    retriever = Mock()
    with pytest.raises(ValueError, match="question_identity"):
        KnowledgeQASkill(retriever).run("owner", "turn", KnowledgeQuery(query="羊毛"),
                                       question=QuestionContext("other", "turn", "cotton"))
    retriever.search.assert_not_called()


def test_mixed_garment_action_only_has_partial_support_without_label():
    result = support_decision("羊毛棉混纺带装饰能直接机洗吗", [], care=True)
    assert result.status == "partial" and not result.hits
    assert result.missing == ("whole_garment_care_label",)


@pytest.mark.parametrize("query", ["羊毛与涤纶混纺怎么洗", "羊毛外套有涂层，如何清洗", "羊毛棉拼接能放进洗衣机吗"])
def test_compound_garment_does_not_need_the_exact_word_machine_wash_to_keep_boundary(query):
    assert support_decision(query, [], care=True).missing == ("whole_garment_care_label",)


def test_selected_item_conflicting_current_material_does_not_choose_either_fibre():
    app = harness()
    reply = app.chat("benchmark-owner", "turn", "这件其实是棉，怎么洗", garment_id="benchmark-wool")
    assert "不一致" in reply.content
    assert reply.sources == []


def test_shakedry_response_keeps_exclusions_without_ordinary_outerwear_permission():
    app = harness()
    reply = app.chat("benchmark-owner", "turn", "SHAKEDRY衣物洗护后可以烘干并补DWR吗")
    assert "SHAKEDRY" in reply.content and "禁止" in reply.content
    assert "部分普通" not in reply.content
    assert "关于 SHAKEDRY 的排除条款" in reply.content
    assert "仅允许相应受热护理" not in reply.content
    assert reply.sources


def test_short_material_does_not_erase_original_operation_from_actual_retrieval():
    app = harness()
    original = app.retriever.search
    queries = []

    def observe(query, *args, **kwargs):
        queries.append(query)
        return original(query, *args, **kwargs)

    app.retriever.search = observe
    reply = app.chat("benchmark-owner", "turn", "梭织羊毛衣物的浮尘应该用衣刷怎么处理")
    assert queries and all("衣刷" in query for query in queries)
    assert "软" in reply.content and "刷" in reply.content
    assert reply.sources


def test_provided_material_without_selected_garment_still_receives_label_clarification():
    app = harness()
    reply = app.chat("benchmark-owner", "turn", "这件三醋酸外套怎么清洗")
    assert "洗标" in reply.content
    assert "请先选择" not in reply.content and reply.sources == []


def test_non_care_knowledge_does_not_offer_a_material_care_tool():
    decision = route_skills("知识库中出差的衣服颜色如何选择", {"care_guide", "rag_search"})
    assert decision.allowed_tools == frozenset({"rag_search"})


def test_explicit_session_document_question_uses_retrieval_without_a_model():
    app = harness()
    reply = app.chat("benchmark-owner", "turn", "根据本会话的Z58资料，该款M码袖长具体多少厘米")
    assert "尺码表" in reply.content and reply.sources == []
    assert any(trace.name == "rag_search" for trace in reply.traces)


def test_generic_storage_prerequisite_is_not_a_general_material_filter_bypass():
    common = {"title": "完全干燥再收纳", "content": "衣物完全干燥后再收纳，不是烘干许可。",
              "topic": "storage", "record_id": "storage-dry-before-fold"}
    assert knowledge_compatible("羊毛收纳前有哪些条件", **common)
    assert not knowledge_compatible("醋酸外套怎么洗", **common)
    assert not knowledge_compatible("羊毛衣物能烘干吗", **common)


def test_storage_answer_combines_scoped_material_and_generic_prerequisite():
    reply = harness().chat("benchmark-owner", "turn", "羊毛衣物长期收纳前应注意什么")
    assert "清洁" in reply.content and "完全干燥" in reply.content
    assert "不是烘干许可" in reply.content
    assert len(reply.sources) == 2
