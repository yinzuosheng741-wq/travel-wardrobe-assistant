import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from evaluation import answer_quality
from evaluation.answer_quality import Review, assess_rules, load_dataset, review_answer, run, summarize


def test_prospective_dataset_fingerprint_and_ids_are_separate_from_regression():
    dataset, manifest = load_dataset()
    critical = json.loads(Path("evaluation/rag_critical_cases.json").read_text(encoding="utf-8"))
    assert len(dataset["cases"]) == manifest["cases"] == 24
    assert {case["id"] for case in dataset["cases"]}.isdisjoint(case["id"] for case in critical["contracts"])
    assert dataset["split"] == "prospective_project_authored"


def test_url_identity_check_is_not_reported_as_semantic_citation_support():
    case = {"citation": "required", "required_any": [["洗标"]]}
    sources = [{"url": "https://example.test/reference"}]
    checks = assess_rules(case, "洗标要求可以直接机洗。", sources, [{"source_url": sources[0]["url"]}])
    assert checks["citation_identity"] is True
    summary = summarize([{"rule_pass": all(checks.values()), "duration_ms": 1, "model_review": None}])
    assert summary["model_assisted_citation_support"]["total"] == 0
    assert summary["human_review"] == "pending_not_scored"


def test_review_schema_does_not_silently_coerce_fabricated_scores():
    with pytest.raises(ValidationError):
        Review.model_validate({"correctness": "100%", "faithfulness": True, "citation_support": None,
                               "clarification_quality": None, "unsupported_claims": [], "reason": ""})


def test_model_assisted_review_rejects_self_inconsistent_grounding_claim():
    class Judge:
        def invoke(self, _messages):
            return AIMessage(content=json.dumps({"correctness": True, "faithfulness": True,
                "citation_support": None, "clarification_quality": True,
                "unsupported_claims": ["unsupported numeric value"], "reason": "inconsistent"}))

    with pytest.raises(ValueError, match="inconsistent_judge"):
        review_answer(Judge(), {"query": "question", "rubric": "reference only"}, "answer", [], [])


def test_existing_human_review_packet_is_never_overwritten(tmp_path):
    output = tmp_path / "report.json"
    template = output.with_suffix(".review-template.json")
    template.write_text("human work", encoding="utf-8")
    with pytest.raises(ValueError, match="retain_existing"):
        run(output)
    assert template.read_text(encoding="utf-8") == "human work"
    assert not output.exists()


def test_frozen_questions_survive_windows_checkout_newline_conversion(tmp_path, monkeypatch):
    path = tmp_path / "cases.json"
    original = answer_quality.DATASET.read_text(encoding="utf-8")
    path.write_bytes(original.replace("\n", "\r\n").encode("utf-8"))
    monkeypatch.setattr(answer_quality, "DATASET", path)
    dataset, _ = load_dataset()
    assert len(dataset["cases"]) == 24
