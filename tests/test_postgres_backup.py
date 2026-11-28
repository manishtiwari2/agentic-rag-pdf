"""Check that failed backups cannot replace an archive."""

import subprocess

import pytest

from scripts import postgres_backup


def test_backup_only_publishes_a_completed_archive(tmp_path, monkeypatch):
    archive = tmp_path / "app.dump"

    def fail(_command, *, stdout, check):
        stdout.write(b"partial")
        assert not archive.exists()
        raise subprocess.CalledProcessError(1, "pg_dump")

    monkeypatch.setattr(postgres_backup.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        postgres_backup.backup(["postgres"], archive)
    assert list(tmp_path.iterdir()) == []

    def succeed(_command, *, stdout, check):
        stdout.write(b"PGDMP")

    monkeypatch.setattr(postgres_backup.subprocess, "run", succeed)
    postgres_backup.backup(["postgres"], archive)
    assert archive.read_bytes() == b"PGDMP"
    with pytest.raises(FileExistsError):
        postgres_backup.backup(["postgres"], archive)
    assert archive.read_bytes() == b"PGDMP"
