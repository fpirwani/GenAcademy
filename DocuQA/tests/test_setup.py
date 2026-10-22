from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

import config
import models


def offline(*args, **kwargs):
    raise httpx.ConnectError("Not running")


def test_missing_ollama_offers_install_in_ui(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(models.httpx, "get", offline)
    monkeypatch.setattr(models, "find_ollama", lambda: None)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    assert not app.exception
    assert any("Would you like to install" in message.value for message in app.warning)
    links = app.get("link_button")
    assert any(
        link.proto.label == "Install Ollama" and link.proto.url == "https://ollama.com/download"
        for link in links
    )
    assert any(button.label == "Check again" for button in app.button)
    assert any(title.value == "Your files" for title in app.subheader)


def test_installed_but_stopped_does_not_offer_reinstall(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(models.httpx, "get", offline)
    monkeypatch.setattr(models, "find_ollama", lambda: Path("ollama.exe"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    assert not app.exception
    assert not app.get("link_button")
    assert any("ollama serve" in code.value for code in app.code)


def test_running_server_needs_no_installer_even_without_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(models, "find_ollama", lambda: None)
    response = httpx.Response(
        200, json={"models": []}, request=httpx.Request("GET", config.OLLAMA_URL)
    )
    monkeypatch.setattr(models.httpx, "get", lambda *a, **kw: response)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    assert not app.exception
    assert not app.get("link_button")
    assert any("ollama pull" in code.value for code in app.code)


def test_check_again_recognizes_completed_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(models.httpx, "get", offline)
    monkeypatch.setattr(models, "find_ollama", lambda: None)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    response = httpx.Response(
        200,
        json={"models": [{"name": config.MODEL}]},
        request=httpx.Request("GET", config.OLLAMA_URL),
    )
    monkeypatch.setattr(models.httpx, "get", lambda *a, **kw: response)
    next(button for button in app.button if button.label == "Check again").click().run()
    assert not app.exception
    assert not app.get("link_button")
    assert models.model_status()[0]


def test_windows_install_is_found_without_refreshed_path(tmp_path, monkeypatch):
    monkeypatch.setattr(models.shutil, "which", lambda _: None)
    monkeypatch.setattr(models.platform, "system", lambda: "Windows")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "programfiles"))
    executable = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    assert models.find_ollama() == executable


def test_missing_embedding_model_offers_setup_without_blocking_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(models, "model_status", lambda: (True, "Local model ready"))
    monkeypatch.setattr(models, "embedding_status", lambda: (False, "Missing"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    assert not app.exception
    assert any("keyword search only" in item.value for item in app.warning)
    assert any("ollama pull nomic-embed-text" in item.value for item in app.code)
