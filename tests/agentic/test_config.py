import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from project_folder.agentic import config


def test_get_providers_in_file_order():
    providers = config.get_providers()
    assert [p["name"] for p in providers] == ["agnes", "opencode", "llm7io", "oraclellm"]


def test_oraclellm_timeouts_read_from_yml():
    providers = config.get_providers()
    oracle = next(p for p in providers if p["name"] == "oraclellm")
    assert oracle["planner_timeout"] == 300.0
    assert oracle["worker_timeout"] == 240.0
    default = next(p for p in providers if p["name"] == "agnes")
    assert default["planner_timeout"] == 60.0
    assert default["worker_timeout"] == 60.0


def test_env_key_format():
    assert config.env_key_for("agnes") == "AGNES_API_KEY"
    assert config.env_key_for("oraclellm") == "ORACLELLM_API_KEY"