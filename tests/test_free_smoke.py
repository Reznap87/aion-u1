import asyncio

import pytest

from aion_u1.models import TeacherResponse
from aion_u1.training.free_smoke import main, smoke, PROMPT, MAX_TOKENS
from aion_u1.training.run_ledger import RunLedger

MODEL = "example/model:free"


class Client:
    def __init__(self, content="AION_OK"):
        self.calls = 0
        self.content = content

    async def require_zero_limit(self):
        pass

    async def list_models(self):
        return [MODEL]

    async def generate(self, model, messages, **kwargs):
        self.calls += 1
        assert model == MODEL
        assert messages == [{"role": "user", "content": PROMPT}]
        assert kwargs["max_tokens"] == MAX_TOKENS
        return TeacherResponse(model_id=model, content=self.content,
                               finish_reason="stop", max_tokens=MAX_TOKENS)


def test_dry_run_does_not_need_key(monkeypatch):
    monkeypatch.delenv("AION_OPENROUTER_FREE_API_KEY", raising=False)
    assert main(["--model", MODEL]) == 0


def test_smoke_once_across_restarts(tmp_path):
    client = Client()
    path = tmp_path / "ledger.sqlite"
    result = asyncio.run(smoke(client, MODEL, RunLedger(path)))
    assert result["passed"] and not result["training_approved"]
    with pytest.raises(RuntimeError, match="replay denied"):
        asyncio.run(smoke(client, MODEL, RunLedger(path)))
    assert client.calls == 1


def test_failed_smoke_not_replayed(tmp_path):
    client = Client("wrong")
    ledger = RunLedger(tmp_path / "ledger.sqlite")
    assert not asyncio.run(smoke(client, MODEL, ledger))["passed"]
    with pytest.raises(RuntimeError, match="replay denied"):
        asyncio.run(smoke(client, MODEL, ledger))
    assert client.calls == 1
