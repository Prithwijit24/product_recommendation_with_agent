"""Load provider/router config from config/api_config.yml (single source of truth)."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "api_config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _defaults() -> dict:
    return load_config().get("defaults") or {}


def get_default(name: str, fallback=None):
    """Read a value from the top-level ``defaults`` section."""
    return _defaults().get(name, fallback)


def get_section(name: str) -> dict:
    """Read a named section from the config (e.g. ``products``, ``research``)."""
    return load_config().get(name) or {}


def env_key_for(provider_name: str) -> str:
    template = get_default("env_key_template", "{name_upper}_API_KEY")
    return template.format(name_upper=provider_name.upper())


def get_providers() -> list[dict]:
    """Providers from the ``providers:`` namespace. Timeouts default to defaults.http_timeout."""
    raw = load_config().get("providers") or {}
    default_timeout = float(get_default("http_timeout", 60.0))
    providers = []
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        providers.append(
            {
                "name": name,
                "order": len(providers),
                "base_url": str(cfg.get("base_url", "")).rstrip("/"),
                "planner_model": cfg.get("planner_model", ""),
                "worker_model": cfg.get("worker_model", ""),
                "planner_timeout": float(cfg.get("planner_timeout", default_timeout)),
                "worker_timeout": float(cfg.get("worker_timeout", default_timeout)),
                "env_key": env_key_for(name),
            }
        )
    return providers


def provider_by_name(name: str) -> dict:
    for p in get_providers():
        if p["name"] == name:
            return p
    raise KeyError(f"No provider named {name!r} in api_config.yml")
