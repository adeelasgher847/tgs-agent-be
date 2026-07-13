from __future__ import annotations

import uuid
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AbTestUpdate(BaseModel):
    """Request body for ``PUT /api/v2/flows/{flow_id}/ab-test``."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    prompt_a_id: uuid.UUID
    prompt_b_id: uuid.UUID
    split_ratio: float = Field(..., ge=0.1, le=0.9)

    @model_validator(mode='after')
    def prompts_must_differ(self) -> AbTestUpdate:
        if self.prompt_a_id == self.prompt_b_id:
            raise ValueError('prompt_a_id and prompt_b_id must be different prompt versions')
        return self


class AbTestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ab_test_enabled: bool
    ab_prompt_a_id: Optional[uuid.UUID]
    ab_prompt_b_id: Optional[uuid.UUID]
    ab_split_ratio: float


class VariantMetrics(BaseModel):
    calls: int
    completed: int
    failed: int
    avg_duration: Optional[float]
    transfer_rate: float
    success_rate: float


class AbResultsResponse(BaseModel):
    variant_a: VariantMetrics
    variant_b: VariantMetrics
    statistical_significance: bool
    recommended_variant: Literal["a", "b", "inconclusive"]


class AbTestWinnerUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: Literal["a", "b"]
