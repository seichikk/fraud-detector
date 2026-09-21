from typing import Any

import pytest
import uvicorn
from pydantic import ValidationError

from fraud_detector import main
from fraud_detector.core.config import Settings, get_settings


def test_run_hands_settings_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)

    main.run()

    assert captured["port"] == get_settings().app_port
    assert captured["log_config"] is None
    assert captured["access_log"] is False


def test_unknown_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")

    with pytest.raises(ValidationError):
        Settings()
