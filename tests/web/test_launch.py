"""The launch entry point composes one local application with explicit settings."""
import sys
from types import ModuleType

import pytest

from agent.web import __main__ as launch
from agent.web.config import WebConfigurationError


def launch_fakes(monkeypatch, *, attachments=False):
    calls = {}
    class Facade:
        resources = object()
    facade = Facade()
    app = object()
    class Configuration:
        input_sets = ("operator-input-set",)
        upload_root = '/workspace/uploads' if attachments else None
        upload_max_bytes = 8192
        upload_max_concurrent = 1
    configuration = Configuration()
    def load(path):
        calls["path"] = path
        return configuration
    def build(value):
        assert value is configuration
        return facade
    web = ModuleType("agent.web.app")
    def create(value, **kwargs):
        assert value is facade
        calls["create"] = kwargs
        return app
    web.create_app = create
    uvicorn = ModuleType("uvicorn")
    def run(value, **kwargs):
        assert value is app
        calls["run"] = kwargs
    uvicorn.run = run
    monkeypatch.setitem(sys.modules, "agent.web.app", web)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setattr(launch, "load_web_configuration", load)
    monkeypatch.setattr(launch, "build_interactive_application", build)
    if attachments:
        from agent.application import uploads
        def admission(resources, root, **kwargs):
            assert resources is facade.resources
            calls['uploads'] = {'root': root, **kwargs}
            return 'upload-owner'
        monkeypatch.setattr(uploads, 'H5ADUploadAdmission', admission)
    return calls


def test_default_launch_is_local_single_process(monkeypatch):
    calls = launch_fakes(monkeypatch)
    assert launch.main(["--config", "operator.json"]) == 0
    assert calls == {
        "path": "operator.json",
        "create": {"input_sets": ("operator-input-set",), "max_workers": 2},
        "run": {"host": "127.0.0.1", "port": 8000, "workers": 1},
    }


def test_explicit_launch_arguments(monkeypatch):
    calls = launch_fakes(monkeypatch)
    assert launch.main(["--config", "operator.json", "--host", "0.0.0.0", "--port", "8123",
                        "--max-workers", "1"]) == 0
    assert calls["run"] == {"host": "0.0.0.0", "port": 8123, "workers": 1}
    assert calls["create"]["max_workers"] == 1


def test_enabled_uploads_use_configured_application_resource_owner(monkeypatch):
    calls = launch_fakes(monkeypatch, attachments=True)
    assert launch.main(['--config', 'operator.json']) == 0
    assert calls['uploads'] == {'root': '/workspace/uploads', 'max_bytes': 8192, 'max_concurrent': 1}
    assert calls['create']['uploads'] == 'upload-owner'
    assert calls['run'] == {'host': '127.0.0.1', 'port': 8000, 'workers': 1}


@pytest.mark.parametrize("arguments", [
    [], ["--config", "operator.json", "--port", "0"],
    ["--config", "operator.json", "--port", "65536"],
    ["--config", "operator.json", "--max-workers", "0"],
    ["--config", "operator.json", "--max-workers", "9"],
])
def test_invalid_launch_arguments_stop_before_server(monkeypatch, arguments):
    calls = launch_fakes(monkeypatch)
    with pytest.raises(SystemExit) as error:
        launch.main(arguments)
    assert error.value.code == 2 and calls == {}


def test_bad_config_reports_startup_error_without_traceback(monkeypatch, capsys):
    calls = launch_fakes(monkeypatch)
    def invalid(path):
        raise WebConfigurationError("Configuration is invalid.")
    monkeypatch.setattr(launch, "load_web_configuration", invalid)
    assert launch.main(["--config", "operator.json"]) == 2
    assert "Configuration is invalid." in capsys.readouterr().err and calls == {}
