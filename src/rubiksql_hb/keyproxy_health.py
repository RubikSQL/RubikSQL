"""Bounded, retrying reads of the proxy's public pool state."""
from __future__ import annotations

from collections.abc import Callable
import http.client
import json
import logging
import time
from typing import Any
import urllib.request

_LOGGER = logging.getLogger(__name__)
HEALTH_TIMEOUT_S = 10.0
HEALTH_ATTEMPTS = 3


def read_health(open_request: Callable[..., Any], url: str, *, timeout: float = HEALTH_TIMEOUT_S) -> dict[str, Any]:
    """Read real pool state, retrying transient transport failures before supervision fails.

    Args:
        open_request: The endpoint's direct, proxy-free request opener.
        url: The already validated proxy origin.
        timeout: Per-attempt socket timeout in seconds.

    Returns:
        The proxy's actual JSON object, including unavailable or paused states.

    Raises:
        OSError: All three transport attempts failed; the final error propagates.
        http.client.HTTPException: All attempts failed to read a complete HTTP response.
        ValueError: The response is invalid JSON or is not an object. Invalid state is never replaced with health.
    """
    for attempt in range(HEALTH_ATTEMPTS):
        try:
            with open_request(urllib.request.Request(url + "/keyproxy/health"), timeout=timeout) as response:
                payload = json.loads(response.read())
        except (OSError, http.client.HTTPException) as error:
            if attempt + 1 == HEALTH_ATTEMPTS:
                raise
            _LOGGER.warning("Proxy pool-state read failed (%s); retrying (%s/%s)",
                            type(error).__name__, attempt + 1, HEALTH_ATTEMPTS)
            time.sleep(.25 * (attempt + 1))
        else:
            if not isinstance(payload, dict):
                raise ValueError("key proxy health must be a JSON object")
            return payload
    raise AssertionError("Health retry loop exhausted without returning or raising")
