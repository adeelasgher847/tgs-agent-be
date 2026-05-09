from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class PromptEngineerRequest(BaseModel):
    """Request schema for the prompt-engineering helper endpoint."""

    requirement: str = Field(
        ...,
        min_length=5,
        max_length=10_000,
        description="User's natural-language requirement for the agent (any language).",
    )
    language_hint: Optional[str] = Field(
        None,
        description="Optional language hint such as 'auto', 'en', 'ur', etc. If omitted, language is auto-detected.",
    )
    tone: Optional[str] = Field(
        None,
        description="Desired tone for the final calling agent, e.g. 'friendly', 'formal', 'salesy'.",
    )
    complexity_level: Optional[str] = Field(
        None,
        description="Audience sophistication, e.g. 'beginner', 'intermediate', 'expert'.",
    )


class PromptEngineerMeta(BaseModel):
    """Auxiliary metadata returned by the prompt engineer model."""

    reasoning_notes: Optional[str] = Field(
        None,
        description="Short internal explanation of assumptions and decisions made while designing the prompt.",
    )


class PromptEngineerResult(BaseModel):
    """Structured result produced by the prompt-engineering model."""

    status: Literal["need_clarification", "ready"] = Field(
        ...,
        description="Whether more information is needed or the prompt is ready to use.",
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="Follow-up questions when requirement is incomplete. Empty when status == 'ready'.",
    )
    final_prompt: Optional[str] = Field(
        None,
        description="The production-ready system prompt for the agent. Null when status == 'need_clarification'.",
    )
    language: str = Field(
        ...,
        description="Detected language code (e.g. 'en', 'ur', 'en-ur').",
    )
    meta: PromptEngineerMeta = Field(
        default_factory=PromptEngineerMeta,
        description="Auxiliary metadata about how the prompt was constructed.",
    )

