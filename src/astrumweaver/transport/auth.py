"""Bearer-token authority boundaries for the v1 transport."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Header, HTTPException, status


@dataclass(frozen=True, slots=True)
class AuthConfig:
    client_token: str
    worker_token: str

    def __post_init__(self) -> None:
        if not self.client_token:
            raise ValueError("client_token must not be empty")
        if not self.worker_token:
            raise ValueError("worker_token must not be empty")
        if secrets.compare_digest(self.client_token, self.worker_token):
            raise ValueError("client and worker tokens must be distinct")


def bearer_guard(expected_token: str):
    async def dependency(authorization: str | None = Header(default=None)) -> None:
        prefix = "Bearer "
        if authorization is None or not authorization.startswith(prefix):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="unauthorized",
            )
        supplied = authorization[len(prefix) :]
        if not secrets.compare_digest(supplied, expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="unauthorized",
            )

    return dependency
