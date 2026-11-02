"""Request-owned, atomic admission shared by all LangGraph branches."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock

_agent = ContextVar("stylemate_agent_role", default="outfit_agent")
_permit = ContextVar("stylemate_tool_permit", default=None)


class BudgetLedger:
    def __init__(self, total=9, per_agent=3):
        self.total, self.per_agent = total, per_agent
        self.tools, self.models, self.rejected = Counter(), Counter(), Counter()
        self.lock = RLock()

    def admit_tool(self, agent):
        with self.lock:
            if sum(self.tools.values()) >= self.total or self.tools[agent] >= self.per_agent:
                self.rejected[agent] += 1
                return False
            self.tools[agent] += 1
            return True

    def admit_model(self, agent):
        with self.lock:
            limit = 1 if agent in {"planner", "synthesizer"} else 4
            if sum(self.models.values()) >= 14 or self.models[agent] >= limit:
                return False
            self.models[agent] += 1
            return True

    def snapshot(self):
        with self.lock:
            return {"tools": dict(self.tools), "models": dict(self.models), "rejected": dict(self.rejected)}


def current_agent():
    return _agent.get()


@contextmanager
def agent_scope(agent):
    token = _agent.set(agent)
    try:
        yield
    finally:
        _agent.reset(token)


@contextmanager
def admitted_tool(name, owner, conversation):
    # A single-use capability, consumed by the bound toolkit before its handler.
    token = _permit.set([name, owner, conversation, False])
    try:
        yield
    finally:
        _permit.reset(token)


def consume_permit(name, owner, conversation):
    permit = _permit.get()
    if permit and permit[:3] == [name, owner, conversation] and not permit[3]:
        permit[3] = True
        return True
    return False
