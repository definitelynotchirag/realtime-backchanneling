"""Shared test isolation.

Tests must not depend on the developer's ``.env``. Without this fixture a
machine with LiveKit fallbacks configured (``LIVEKIT_URL_2``, ...) would make
tests probe real projects, and the observed project would change with whatever
credentials happen to be lying around.
"""

import os

import pytest

ISOLATED_PREFIXES = (
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "LIVEKIT_AGENT_NAME",
    "LIVEKIT_STATE_PATH",
)


@pytest.fixture(autouse=True)
def livekit_environment_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without ambient LiveKit configuration."""

    for name in [name for name in os.environ if name.startswith(ISOLATED_PREFIXES)]:
        monkeypatch.delenv(name, raising=False)
