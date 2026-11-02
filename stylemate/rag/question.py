"""Immutable turn constraints, bound by trusted entry points rather than tool arguments."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class QuestionContext:
    owner_id: str
    conversation_id: str
    original_query: str
    garment_id: str | None = None
    recorded_material: str | None = None

    def for_garment(self, garment_id: str, material: str | None):
        return replace(self, garment_id=garment_id, recorded_material=material)

    @property
    def constraint_query(self):
        if self.recorded_material:
            return f"{self.original_query}\n用户记录的衣物材质：{self.recorded_material}"
        return self.original_query


_question: ContextVar[QuestionContext | None] = ContextVar("stylemate_question", default=None)


def current_question(owner_id: str, conversation_id: str) -> QuestionContext | None:
    question = _question.get()
    if question and (question.owner_id, question.conversation_id) == (owner_id, conversation_id):
        return question
    return None


@contextmanager
def question_scope(question: QuestionContext):
    token = _question.set(question)
    try:
        yield question
    finally:
        _question.reset(token)
