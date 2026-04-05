"""Provider registry — discover and instantiate ingestion providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cortex.providers.base import IngestProvider

_PROVIDER_FACTORIES: dict[str, type] = {}


def _ensure_registered():
    """Lazily import provider classes so they register themselves."""
    if _PROVIDER_FACTORIES:
        return
    from cortex.providers.claude import ClaudeHistoryProvider  # noqa: F401
    from cortex.providers.goose import GooseProvider  # noqa: F401


def register(name: str, cls: type) -> None:
    """Register a provider class under a name."""
    _PROVIDER_FACTORIES[name] = cls


def get_provider(name: str, **kwargs) -> IngestProvider:
    """Get a provider instance by name."""
    _ensure_registered()
    cls = _PROVIDER_FACTORIES.get(name)
    if cls is None:
        raise ValueError(f"Unknown provider: {name}. Available: {list(_PROVIDER_FACTORIES.keys())}")
    return cls(**kwargs)


def discover_providers() -> list[str]:
    """Return names of providers whose data sources exist on this machine."""
    _ensure_registered()
    available = []
    for name, cls in _PROVIDER_FACTORIES.items():
        try:
            instance = cls()
            if instance.detect():
                available.append(name)
        except Exception:
            pass
    return available
