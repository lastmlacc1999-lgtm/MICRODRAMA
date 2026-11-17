"""Tests must never load the user's saved keys or write to their projects."""
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Pytest loads conftest before importing test modules (some import api at module
# scope). Point dotenv away from server/.env before api.py can load it.
_test_env_dir = Path(tempfile.mkdtemp(prefix="microdrama-test-env-"))
os.environ["MICRODRAMA_ENV_FILE"] = str(_test_env_dir / ".env")
for _name in (
    "A2E_API_KEY", "A2E_BASE_URL", "A2E_IMAGE_MODEL", "A2E_VIDEO_MODEL", "A2E_AUTO_ADD_AUDIO",
    "A2E_MAX_JOB_CREDITS", "A2E_MAX_CONCURRENCY", "A2E_POLL_INTERVAL_SECONDS", "A2E_POLL_TIMEOUT_SECONDS",
    "GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "MAX_SCENES", "MAX_SHOTS_PER_SCENE",
    "MAX_CHARACTERS",
):
    os.environ.pop(_name, None)


@pytest.fixture(autouse=True)
def isolated_api_storage(monkeypatch, tmp_path):
    import api
    monkeypatch.setattr(api, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(api, "LEGACY_OUTPUTS_DIR", tmp_path / "outputs")
    api.jobs.clear()
    yield
    api.jobs.clear()


def pytest_unconfigure(config):
    shutil.rmtree(_test_env_dir, ignore_errors=True)
