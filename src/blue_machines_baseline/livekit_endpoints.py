"""LiveKit project failover.

The worker's registration, the browser join token, and the scripted simulator
must all point at the same LiveKit project, or rooms get created in a project
with no worker attached. Projects are listed as ``LIVEKIT_URL`` /
``LIVEKIT_API_KEY`` / ``LIVEKIT_API_SECRET`` plus numbered fallbacks (``_2``,
``_3``); the first project that answers a probe call wins, and the choice is
cached in a small state file so the API and the worker converge on it without
probing on every request. An expired-credit project simply fails the probe and
the next one takes over.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from livekit import api as livekit_api

from .config import ConfigurationError, Settings

logger = logging.getLogger("blue-machines-livekit")

PROBE_TIMEOUT_SECONDS: float = 4.0
"""How long one project probe may take before the project counts as unavailable."""

STATE_TTL_SECONDS: float = 45.0
"""How long a recorded choice is trusted before the projects are probed again."""

MAX_FALLBACKS: int = 8
"""Upper bound on numbered fallbacks (LIVEKIT_URL_2 ... LIVEKIT_URL_9)."""

DEFAULT_STATE_PATH = Path(".runtime/livekit-active.json")


@dataclass(frozen=True)
class LiveKitEndpoint:
    """One LiveKit project's connection triplet, in preference order."""

    index: int
    url: str
    api_key: str
    api_secret: str

    @property
    def label(self) -> str:
        return "primary" if self.index == 0 else f"fallback #{self.index}"


def endpoints_from_env(values: Mapping[str, str] | None = None) -> list[LiveKitEndpoint]:
    """Read the ordered project list from the environment.

    The primary is unnumbered; fallbacks are ``_2``, ``_3``, ... A number that
    appears partially (say a URL without its key) is a configuration error
    rather than a silently ignored entry.
    """

    source = os.environ if values is None else values
    endpoints: list[LiveKitEndpoint] = []
    for index in range(MAX_FALLBACKS + 1):
        suffix = "" if index == 0 else f"_{index + 1}"
        url = source.get(f"LIVEKIT_URL{suffix}", "").strip()
        api_key = source.get(f"LIVEKIT_API_KEY{suffix}", "").strip()
        api_secret = source.get(f"LIVEKIT_API_SECRET{suffix}", "").strip()
        if not any((url, api_key, api_secret)):
            continue
        missing = [
            name
            for name, value in (
                (f"LIVEKIT_URL{suffix}", url),
                (f"LIVEKIT_API_KEY{suffix}", api_key),
                (f"LIVEKIT_API_SECRET{suffix}", api_secret),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                f"Incomplete LiveKit configuration for {suffix or 'the primary'}: "
                f"missing {', '.join(missing)}"
            )
        parsed = urlparse(url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            raise ConfigurationError(f"LIVEKIT_URL{suffix} must be a valid ws:// or wss:// URL")
        endpoints.append(LiveKitEndpoint(index, url, api_key, api_secret))
    if not endpoints:
        raise ConfigurationError("Missing required environment variable(s): LIVEKIT_URL")
    return endpoints


def _state_path(values: Mapping[str, str] | None) -> Path:
    source = os.environ if values is None else values
    configured = source.get("LIVEKIT_STATE_PATH", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_STATE_PATH


def _probe(endpoint: LiveKitEndpoint, timeout: float = PROBE_TIMEOUT_SECONDS) -> bool:
    """Ask the project one cheap question; anything but success is a failure."""

    async def probe_once() -> None:
        client = livekit_api.LiveKitAPI(
            url=endpoint.url.replace("wss://", "https://").replace("ws://", "http://"),
            api_key=endpoint.api_key,
            api_secret=endpoint.api_secret,
        )
        try:
            await client.room.list_rooms(livekit_api.ListRoomsRequest())
        finally:
            await client.aclose()

    async def guarded() -> None:
        await asyncio.wait_for(probe_once(), timeout=timeout)

    try:
        asyncio.run(guarded())
    except Exception as exc:  # noqa: BLE001 - any failure means "not this project"
        logger.warning(
            "livekit project %s (%s) is unavailable: %s: %s",
            endpoint.label,
            endpoint.url,
            type(exc).__name__,
            exc,
        )
        return False
    return True


def _read_state(path: Path) -> dict[str, object] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_state(path: Path, endpoint: LiveKitEndpoint, checked_at: float) -> None:
    payload = json.dumps(
        {"index": endpoint.index, "url": endpoint.url, "checked_at": checked_at},
        separators=(",", ":"),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    except OSError:
        logger.warning("could not persist the LiveKit project choice to %s", path)


def resolve_active_endpoint(
    values: Mapping[str, str] | None = None,
    *,
    state_path: Path | None = None,
    ttl_seconds: float = STATE_TTL_SECONDS,
    prober: Callable[[LiveKitEndpoint], bool] | None = None,
    clock: Callable[[], float] | None = None,
) -> LiveKitEndpoint | None:
    """Return the first reachable project, caching the choice between probes.

    With a single configured project there is nothing to fail over to, so it is
    returned without a probe. With fallbacks present, a recent cached choice is
    trusted for ``ttl_seconds``; after that (or with no cache) projects are
    probed in preference order and the first success is recorded. ``None`` means
    no project answered.
    """

    endpoints = endpoints_from_env(values)
    if len(endpoints) == 1:
        return endpoints[0]

    path = state_path if state_path is not None else _state_path(values)
    now = (clock or time.time)()
    probe = prober or _probe

    state = _read_state(path)
    if state is not None:
        try:
            cached_index = int(state.get("index", -1))  # type: ignore[arg-type]
            checked_at = float(state.get("checked_at", 0.0))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            cached_index, checked_at = -1, 0.0
        if now - checked_at < ttl_seconds:
            for endpoint in endpoints:
                if endpoint.index == cached_index:
                    return endpoint

    for endpoint in endpoints:
        if probe(endpoint):
            if endpoint.index != 0:
                logger.warning(
                    "using livekit %s (%s); earlier projects did not answer",
                    endpoint.label,
                    endpoint.url,
                )
            _write_state(path, endpoint, now)
            return endpoint

    logger.error("no configured livekit project answered a probe")
    return None


def apply_endpoint(settings: Settings) -> tuple[Settings, LiveKitEndpoint | None]:
    """Resolve the active project and fold it back into the worker's settings."""

    endpoint = resolve_active_endpoint()
    if endpoint is None:
        logger.error("keeping the primary LiveKit project: no project answered a probe")
        return settings, None
    if endpoint.url == settings.livekit_url:
        return settings, endpoint
    logger.warning("switching this process to livekit %s (%s)", endpoint.label, endpoint.url)
    updated = settings.model_copy(
        update={
            "livekit_url": endpoint.url,
            "livekit_api_key": type(settings.livekit_api_key)(endpoint.api_key),
            "livekit_api_secret": type(settings.livekit_api_secret)(endpoint.api_secret),
        }
    )
    return updated, endpoint
