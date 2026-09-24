"""Behaviour of the LiveKit project failover.

The point of the failover is that the worker, the browser token, and the
simulator agree on one project: if these tests drift, rooms can be created in a
project the worker is not registered to.
"""

from pathlib import Path

import pytest

from blue_machines_baseline.config import ConfigurationError
from blue_machines_baseline.livekit_endpoints import (
    LiveKitEndpoint,
    endpoints_from_env,
    resolve_active_endpoint,
)


def environment_with(*project_count: int) -> dict[str, str]:
    """Environment where projects 1..n are configured (1 is the primary)."""

    total = max(project_count)
    values: dict[str, str] = {}
    for index in range(1, total + 1):
        suffix = "" if index == 1 else f"_{index}"
        values[f"LIVEKIT_URL{suffix}"] = f"wss://project{index}.livekit.cloud"
        values[f"LIVEKIT_API_KEY{suffix}"] = f"key-{index}"
        values[f"LIVEKIT_API_SECRET{suffix}"] = f"secret-{index}"
    return values


def test_endpoints_follow_the_numbered_preference_order() -> None:
    endpoints = endpoints_from_env(environment_with(1, 2, 3))

    assert [(item.index, item.url) for item in endpoints] == [
        (0, "wss://project1.livekit.cloud"),
        (1, "wss://project2.livekit.cloud"),
        (2, "wss://project3.livekit.cloud"),
    ]


def test_incomplete_fallback_triplet_is_a_configuration_error() -> None:
    values = environment_with(1, 2)
    values.pop("LIVEKIT_API_SECRET_2")

    with pytest.raises(ConfigurationError, match="LIVEKIT_API_SECRET_2"):
        endpoints_from_env(values)


def test_single_project_is_used_without_probing(tmp_path: Path) -> None:
    def probe(endpoint: LiveKitEndpoint) -> bool:
        raise AssertionError("a single configured project must not be probed")

    endpoint = resolve_active_endpoint(
        environment_with(1), state_path=tmp_path / "state.json", prober=probe
    )

    assert endpoint is not None and endpoint.url == "wss://project1.livekit.cloud"


def test_failed_primary_advances_to_the_next_project_and_is_cached(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    probed: list[int] = []

    def probe(endpoint: LiveKitEndpoint) -> bool:
        probed.append(endpoint.index)
        return endpoint.index == 1

    endpoint = resolve_active_endpoint(
        environment_with(1, 2, 3), state_path=state_path, prober=probe, clock=lambda: 100.0
    )

    assert endpoint is not None and endpoint.url == "wss://project2.livekit.cloud"
    assert probed == [0, 1]  # the primary was tried first and failed

    def probe_again(endpoint: LiveKitEndpoint) -> bool:
        raise AssertionError("a fresh cached choice must not be re-probed")

    cached = resolve_active_endpoint(
        environment_with(1, 2, 3),
        state_path=state_path,
        prober=probe_again,
        clock=lambda: 120.0,
    )

    assert cached is not None and cached.url == "wss://project2.livekit.cloud"


def test_expired_choice_reprobes_and_returns_to_a_recovered_primary(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"

    resolve_active_endpoint(
        environment_with(1, 2),
        state_path=state_path,
        prober=lambda endpoint: endpoint.index == 1,
        clock=lambda: 100.0,
        ttl_seconds=30.0,
    )
    probed: list[int] = []

    def probe(endpoint: LiveKitEndpoint) -> bool:
        probed.append(endpoint.index)
        return True  # the primary has recovered

    endpoint = resolve_active_endpoint(
        environment_with(1, 2),
        state_path=state_path,
        prober=probe,
        clock=lambda: 131.0,
        ttl_seconds=30.0,
    )

    assert endpoint is not None and endpoint.url == "wss://project1.livekit.cloud"
    assert probed == [0]


def test_no_reachable_project_returns_none(tmp_path: Path) -> None:
    endpoint = resolve_active_endpoint(
        environment_with(1, 2, 3),
        state_path=tmp_path / "state.json",
        prober=lambda _endpoint: False,
        clock=lambda: 5.0,
    )

    assert endpoint is None
