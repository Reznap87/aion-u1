import pytest

from aion_u1.training.model_identity import artifact_digest, model_identity, validate_identity


def test_local_artifact_digest_changes_with_content_and_name(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"weights")
    first = artifact_digest(tmp_path)
    assert artifact_digest(tmp_path) == first
    path.write_bytes(b"new weights")
    second = artifact_digest(tmp_path)
    assert second != first
    path.rename(tmp_path / "renamed.safetensors")
    assert artifact_digest(tmp_path) != second


def test_pinned_hub_identity():
    revision = "a" * 40
    identity = model_identity("example/model", revision, None)
    assert validate_identity(identity, "example/model", None) == identity
    with pytest.raises(ValueError, match="commit SHA"):
        model_identity("example/model", "main", None)


def test_adapter_identity_changes(tmp_path):
    path = tmp_path / "adapter.safetensors"
    path.write_bytes(b"first")
    first = model_identity("example/model", "a" * 40, str(tmp_path))
    path.write_bytes(b"second")
    second = model_identity("example/model", "a" * 40, str(tmp_path))
    assert first["adapter"] != second["adapter"]


def test_empty_artifact_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        artifact_digest(tmp_path)


def test_unpinned_model_cannot_be_compared():
    identity = model_identity("example/model", None, None)
    with pytest.raises(ValueError, match="immutable identity"):
        validate_identity(identity, "example/model", None)
