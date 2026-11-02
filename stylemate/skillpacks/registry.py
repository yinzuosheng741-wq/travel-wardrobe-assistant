"""Explicit discovery and typed execution of packaged Skills (MediX design)."""
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from stylemate.agent.budget import admitted_tool
from stylemate.agent.policy import denied_tool_payload
from stylemate.agent.routing import WRITE_TOOLS
from stylemate.agent.tool_results import error_payload
from stylemate.constraints import load_capabilities


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(ge=1, le=1)
    skills: dict[str, list[str]]


class SkillRegistry:
    def __init__(self, tools):
        self.manifest = Manifest.model_validate(yaml.safe_load(Path(__file__).with_name("manifest.yaml").read_text("utf-8")))
        names = [name for values in self.manifest.skills.values() for name in values]
        if len(set(names)) != len(names) or len(self.manifest.skills) != 7:
            raise ValueError("invalid_skill_catalog")
        self.descriptions = {}
        for skill in self.manifest.skills:
            document = (Path(__file__).parent / skill / "SKILL.md").read_text("utf-8")
            metadata = yaml.safe_load(document.split("---", 2)[1])
            if metadata.get("name") != skill or not metadata.get("description"):
                raise ValueError("invalid_skill_metadata")
            self.descriptions[skill] = metadata["description"]
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools) or set(self.tools) - set(names):
            raise ValueError("unregistered_skill_function")
        self.capabilities = load_capabilities()
        for capability in self.capabilities.agents.values():
            if set(capability.skills) - set(self.manifest.skills) or set(capability.functions) - set(names):
                raise ValueError("invalid_skill_reference")

    def allowed(self, agent, request_tools):
        capability = self.capabilities.agents[agent]
        names = set(capability.functions)
        for skill in capability.skills:
            names.update(self.manifest.skills[skill])
        # Only the management agent may prepare writes; request authorization
        # remains enforced independently by the bound toolkit.
        if agent != "wardrobe_agent":
            names -= WRITE_TOOLS
        return names & set(request_tools) & set(self.tools)

    def visible(self, agent, request_tools):
        return [self.tools[name] for name in sorted(self.allowed(agent, request_tools))]

    def execute(self, agent, allowed, name, arguments, owner, conversation):
        denied = denied_tool_payload(name, owner, conversation)
        if denied:
            return json.loads(denied)
        if name not in self.allowed(agent, allowed):
            return self.failure(name, "skill_permission_denied")
        try:
            if not isinstance(arguments, dict) or set(arguments) - set(self.tools[name].args_schema.model_fields):
                raise ValueError("undeclared_arguments")
            with admitted_tool(name, owner, conversation):
                return json.loads(self.tools[name].invoke(arguments))
        except Exception:
            return self.failure(name, "invalid_arguments_or_tool_failure")

    @staticmethod
    def failure(name, reason):
        return {**error_payload("工具未完成，请检查条件或缩小问题后重试。", reason),
                "trace": {"name": name or "unknown", "status": "failed", "summary": reason, "duration_ms": 0}}
