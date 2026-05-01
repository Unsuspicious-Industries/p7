import sys
import types

import p7.modal_deployment as modal_module
from p7.modal_deployment import ModalDeployment, load_modal_env


def test_load_modal_env_reads_modal_credentials(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "MODAL_TOKEN_ID=token-id\nMODAL_TOKEN_SECRET='token-secret'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    loaded = load_modal_env(env_path)

    assert loaded == {
        "MODAL_TOKEN_ID": "token-id",
        "MODAL_TOKEN_SECRET": "token-secret",
    }


def test_modal_deployment_uses_env_file_credentials(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "MODAL_TOKEN_ID=token-id\nMODAL_TOKEN_SECRET=token-secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    deployment = ModalDeployment("gpt2", "stlc", env_path=env_path)

    assert deployment.token_id == "token-id"
    assert deployment.token_secret == "token-secret"


def test_modal_deployment_calls_remote_function(monkeypatch):
    calls = []

    class FakeRemoteFunction:
        def with_options(self, **kwargs):
            calls.append(("with_options", kwargs))
            return self

        def remote(self, **kwargs):
            calls.append(("remote", kwargs))
            return {
                "text": "typed-output",
                "is_complete": True,
                "tokens_generated": 2,
                "stopped_reason": "complete",
                "tokens": ["typed", "-output"],
            }

    class FakeFunction:
        @staticmethod
        def from_name(app_name, function_name):
            calls.append(("from_name", app_name, function_name))
            return FakeRemoteFunction()

    fake_modal = types.SimpleNamespace(Function=FakeFunction)
    monkeypatch.setattr(modal_module, "_modal", fake_modal)

    deployment = ModalDeployment("gpt2", "stlc", gpu="T4")
    result = deployment.generate_constrained(
        "prompt",
        initial="λx:Int.",
        max_tokens=8,
    )

    assert result.text == "typed-output"
    assert result.is_complete is True
    assert calls[0] == ("from_name", "proposition7-generation", "generate_constrained")
    assert calls[1] == ("with_options", {"gpu": "T4"})
    assert calls[2][0] == "remote"
    assert calls[2][1]["model_name"] == "gpt2"
    assert calls[2][1]["grammar"] == "stlc"


def test_modal_deployment_deploy_uses_modal_runner(monkeypatch):
    deployed = []
    runner = types.ModuleType("modal.runner")

    def deploy_app(app, name=None):
        deployed.append((app, name))

    runner.deploy_app = deploy_app
    fake_app = object()
    fake_modal = types.SimpleNamespace(Function=object)

    monkeypatch.setitem(sys.modules, "modal", types.ModuleType("modal"))
    monkeypatch.setitem(sys.modules, "modal.runner", runner)
    monkeypatch.setattr(modal_module, "_modal", fake_modal)
    monkeypatch.setattr(modal_module, "app", fake_app)

    deployment = ModalDeployment("gpt2", "stlc", app_name="custom-app")

    assert deployment.deploy() is deployment
    assert deployed == [(fake_app, "custom-app")]
