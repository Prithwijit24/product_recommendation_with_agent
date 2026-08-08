"""Load provider/router config from config/api_config.yml (single source of truth)."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "api_config.yml"

DEFAULT_TIMEOUT = 60.0


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def env_key_for(provider_name: str) -> str:
    return f"{provider_name.upper()}_API_KEY"


def get_providers() -> list[dict]:
    """Providers in yaml file order. Timeouts default to 60s unless yml overrides."""
    raw = load_config()
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
                "planner_timeout": float(cfg.get("planner_timeout", DEFAULT_TIMEOUT)),
                "worker_timeout": float(cfg.get("worker_timeout", DEFAULT_TIMEOUT)),
                "env_key": env_key_for(name),
            }
        )
    return providers


def provider_by_name(name: str) -> dict:
    for p in get_providers():
        if p["name"] == name:
            return p
    raise KeyError(f"No provider named {name!r} in api_config.yml")