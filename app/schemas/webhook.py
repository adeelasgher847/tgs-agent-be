from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator

from app.utils.ssrf import assert_public_url


class WebhookEndpointCreate(BaseModel):
    url: AnyHttpUrl
    secret: str = Field(min_length=16)

    @field_validator("url")
    @classmethod
    def must_be_https_and_public(cls, v: AnyHttpUrl) -> AnyHttpUrl:
        url_str = str(v)
        if url_str.startswith("http://"):
            raise ValueError("Webhook URL must use HTTPS")
        # SSRF guard — raises SSRFBlockedError (ValueError subclass) which Pydantic
        # converts to a validation error message.
        assert_public_url(url_str)
        return v


class WebhookEndpointOut(BaseModel):
    id: uuid.UUID
    url: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class WebhookDeliveryOut(BaseModel):
    id: uuid.UUID
    endpoint_id: uuid.UUID
    event_type: str
    payload: dict
    status: str
    http_status: Optional[int]
    response_body: Optional[str]
    attempt_count: int
    last_attempted_at: datetime
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedWebhookDeliveries(BaseModel):
    items: list[WebhookDeliveryOut]
    total: int
    page: int
    page_size: int
