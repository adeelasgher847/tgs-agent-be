from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, model_validator


class SlackAuthorizeResponse(BaseModel):
    authorization_url: str


class SlackChannelOut(BaseModel):
    id: str
    name: str


class SlackDisconnectResponse(BaseModel):
    disconnected: bool
    provider: str = "slack"


class SlackIntegrationStatusOut(BaseModel):
    connected: bool
    connected_at: datetime | None = None
    team_name: str | None = None
    default_channel_id: str | None = None
    default_channel_name: str | None = None


class SlackSetChannelRequest(BaseModel):
    channel_id: str | None = None
    channel_name: str | None = None

    @model_validator(mode="after")
    def validate_channel_pair(self) -> "SlackSetChannelRequest":
        if bool(self.channel_id) != bool(self.channel_name):
            raise ValueError(
                "channel_id and channel_name must both be set (to select a "
                "default channel) or both be omitted/null (to clear it)"
            )
        return self
