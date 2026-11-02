"""Validated, packaged capabilities. Configuration errors fail closed."""
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AgentCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    skills: list[str]
    functions: list[str] = Field(default_factory=list)
    max_tool_calls: int = Field(ge=3, le=3)


class Capabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(ge=1, le=1)
    agents: dict[str, AgentCapability]
    output_rules: list[str]


@lru_cache(maxsize=1)
def load_capabilities():
    result = Capabilities.model_validate(yaml.safe_load(Path(__file__).with_name("agents.yaml").read_text("utf-8")))
    if set(result.agents) != {"wardrobe_agent", "outfit_agent", "knowledge_agent"}:
        raise ValueError("invalid_agent_catalog")
    if set(result.output_rules) != {"preserve_care_boundaries", "inventory_ids_must_exist", "writes_require_confirmation"}:
        raise ValueError("invalid_output_rules")
    return result
